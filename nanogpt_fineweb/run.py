import os
import time
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
import torch._inductor.config as torch_config
from types import SimpleNamespace


import torch._dynamo
torch._dynamo.config.suppress_errors = True


from layers.astralora_layer import AstraloraLayer


from config.config import config
from utils.utils import init_log
from utils.utils import init_neptune
from utils.utils import init_path
from utils.utils import init_seed
from utils.utils import modify_gpu_args_for_cryri
from utils.utils import save_args_to_markdown


from data import DistributedDataLoader
from model import Model
from optimizer_muon import OptimizerMuon


@record
@torch.compiler.disable
def run(args, args_parser):
    # --- Set the global seed value
    init_seed(args.seed)

    # --- Set up DDP (distributed data parallel) computation mode:
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    master_process = (ddp_rank == 0)
    
    # --- Init device:
    assert torch.cuda.is_available(), 'It works only with cuda'
    gpu_num = int(args.gpus.replace(' ', '').split(',')[ddp_local_rank])
    device = f'cuda:{gpu_num}'
    torch.cuda.set_device(device)

    # --- Init folders and log:
    if master_process:
        init_path(args.name, args.root, args.rewrite)

    args.folder = os.path.join(args.root, args.name)
    fpath = os.path.join(args.folder, 'log.txt')
    log = init_log(fpath=fpath, enable=master_process)

    if master_process:
        fpath = os.path.join(args.folder, 'args.md')
        save_args_to_markdown(args, args_parser, fpath)

    # --- Initialize Neptune:
    if master_process:
        nepman, url = init_neptune(args.name, 'set_neptune_env.sh', args)
        log('Use neptune. See: ' + url, 'res')
    else:
        nepman = None

    # --- Calculate the number of steps to take in the validation loop:
    B = args.batch_size
    T = args.sequence_length
    assert args.tokens_vld % (B * T * ddp_world_size) == 0
    val_steps = args.tokens_vld // (B * T * ddp_world_size)
    
    # --- Prepare data:
    loader_trn = DistributedDataLoader(args, ddp_rank, ddp_world_size)
    loader_vld = DistributedDataLoader(args, ddp_rank, ddp_world_size, vld=True)
    
    # --- Log info:
    if master_process:
        nepman['info/ddp_world_size'] = ddp_world_size
        nepman['info/data_trn'] = loader_trn.info()
        nepman['info/data_vld'] = loader_vld.info()
        
    x, y = loader_trn.next_batch()

    # --- Init the model:
    model = Model(SimpleNamespace(
        vocab_size=50304, block_size=1024,
        n_layer=args.num_blocks, n_head=args.num_head, n_embd=768*2,
        mode=args.mode, bb_d=args.bb_d, bb_kind=args.bb_kind,
        rank=args.rank, samples_bb=args.samples_bb, samples_sm=args.samples_sm,
        use_sm=not args.use_stochastic_w,
        use_gd_update=args.use_gd_update,
        gd_update_iters=args.gd_update_iters,
        log=log, nepman=nepman))
    model = model.cuda()
    model.master_process = master_process
    
    # --- Special trick for speeding up:
    if hasattr(torch_config, 'coordinate_descent_tuning'):
        torch_config.coordinate_descent_tuning = True # suggested by @Chillee

    model = torch.compile(model)
    
    # --- Wrap model into DDP container:
    model = DDP(model, device_ids=[gpu_num])
    model_raw = model.module # Always contains the unwrapped model
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float32)

    # --- Collect parameters for training proccess:
    params1 = model_raw.get_head_params()
    params2 = [] # model_raw.transformer.h.parameters()
    params3 = []
    bb_param_ids = set()
    for module in model_raw.transformer.h.modules():
        if hasattr(module, 'bb_wrapper'):
            for param in module.parameters():
                bb_param_ids.add(id(param))
    for param in model_raw.transformer.h.parameters():
        if id(param) in bb_param_ids:
            params3.append(param)
        else:
            params2.append(param)

    # --- Init the optimizers:
    optimizer1 = torch.optim.AdamW(params1,
        lr=args.lr_embed,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=True)
    optimizer2 = OptimizerMuon(params2,
        lr=args.lr_muon,
        momentum=0.95,
        rank=ddp_rank,
        world_size=ddp_world_size)
    optimizers = [optimizer1, optimizer2]
    if len(params3) > 0:
        optimizer3 = torch.optim.AdamW(params3,
            lr=args.lr_bb,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
            fused=True)
        optimizers.append(optimizer3)
    
    # --- Set the learning rate decay scheduler (linear warmup and warmdown):
    def get_lr(it):
        assert it <= args.num_iterations
        # 1) linear warmup for warmup_iters steps:
        if it < args.warmup_iters:
            return (it+1) / args.warmup_iters
        # 2) constant lr for a while:
        elif it < args.num_iterations - args.warmdown_iters:
            return 1.0
        # 3) linear warmdown:
        else:
            decay_ratio = (args.num_iterations - it) / args.warmdown_iters
            return decay_ratio
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr)
        for opt in optimizers]

    # --- Start the clock:
    training_time_ms = 0
    torch.cuda.synchronize()
    t0 = time.time()

    # --- Training loop:
    loader_trn.reset()
    for step in range(args.num_iterations + 1):
        last_step = (step == args.num_iterations)

        if step == 10:
            # We do not count first 10 steps for timing, which are slow
            training_time_ms = 0
            t0 = time.time()
        timed_steps = float('nan') if step <= 11 else (step - 10) + 1

        # --- Evaluate the validation dataset:
        do_vld = args.vld_every > 0 and step % args.vld_every == 0
        do_vld = do_vld or last_step
        if do_vld:
            # Stop the clock:
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
            
            # Run validation batches:
            model.eval()
            loader_vld.reset()
            loss_vld = 0.
            for _ in range(val_steps):
                x_val, y_val = loader_vld.next_batch()
                with ctx:
                    _, loss = model(x_val, y_val, return_logits=False)
                    loss_vld += loss.detach()
                    del loss
            dist.all_reduce(loss_vld, op=dist.ReduceOp.AVG)
            loss_vld /= val_steps
            
            # Log the value:
            if master_process:
                log(f'Validation # {step+1:-4d} | loss {loss_vld:.4f}', 'res')
                nepman['validation/loss'].append(loss_vld.item())
                nepman['validation/step'].append(step + 1)

            # Start the clock again:
            torch.cuda.synchronize()
            t0 = time.time()

        if last_step:
            break

        # --- Train the model:
        model.train()
        for i in range(1, args.accumulation_steps+1):
            # Forward pass:
            with ctx:
                _, loss = model(x, y, return_logits=False)
                loss_trn = loss.detach()
            # Advance the dataset for the next batch:
            x, y = loader_trn.next_batch()
            # Backward pass:
            if i < args.accumulation_steps:
                with model.no_sync():
                    loss.backward()
            else:
                loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= args.accumulation_steps
        
        # --- Step the optimizers and schedulers:
        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()
        model.zero_grad(set_to_none=True)

        # --- Log the train results:
        if master_process:
            approx_time = training_time_ms + 1000 * (time.time() - t0)
            log(f'step:{step+1}/{args.num_iterations} train_loss:{loss_trn.item():.4f} train_time:{approx_time/1000:.2f}s step_avg:{approx_time/timed_steps/1000:.2f}s')
            nepman['training/step'].append(step + 1)
            nepman['training/loss'].append(loss_trn.item())
            nepman['training/time'].append(approx_time / 1000)
            nepman['training/step_time'].append(approx_time/timed_steps/1000)

    # --- Save the trained model:
    if master_process and args.save_model:
        # Stop the clock:
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        # Save the state of the training process:
        fpath = os.path.join(args.root, args.name, 'model.pt')
        torch.save(model_raw.state_dict(), fpath)
        # Start the clock again:
        torch.cuda.synchronize()
        t0 = time.time()

    # --- Completion of the work process:
    if master_process:
        mem = torch.cuda.max_memory_allocated() // 1024 // 1024
        log(f'Memory used: {mem}', 'res')
        nepman['system/memory_used'] = mem
        nepman.stop()
        
    dist.destroy_process_group()


if __name__ == '__main__':
    args, args_parser = config('nanogpt_fineweb')
    args = modify_gpu_args_for_cryri(args)

    run(args, args_parser)