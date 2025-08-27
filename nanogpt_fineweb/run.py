"""Train the GPT2-like model on FineWeb dataset.

See:
- https://github.com/KellerJordan/modded-nanogpt
- https://github.com/karpathy/nanoGPT

Usage:
1. Donwload data: "cd nanogpt_fineweb && python run_data.py"
2. Run: "torchrun --standalone --nproc_per_node=1 nanogpt_fineweb/run.py"

"""
import os
from time import perf_counter as tpc
import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
import torch._inductor.config as torch_config
from types import SimpleNamespace


import torch._dynamo
torch._dynamo.config.suppress_errors = True


from core.astralora import Astralora


from data import DistributedDataLoader
from model import Model
from optimizer_muon import OptimizerMuon


@record
@torch.compiler.disable
def run():
    # --- Set up DDP (distributed data parallel) computation mode:
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    master_process = (ddp_rank == 0)
    
    # TODO: add support for multi-gpu (if master_process)
    ast = Astralora('nanogpt_fineweb', master_process=master_process)
    args = ast.args

    # --- Init device:
    assert torch.cuda.is_available(), 'It works only with cuda'
    
    if args.device_total > 1:
        gpu_num = ddp_local_rank
        torch.cuda.set_device(f'cuda:{gpu_num}')
    else:
        gpu_num = ast.device
        torch.cuda.set_device(ast.device)
        
    # --- Calculate the number of steps to take in the validation loop:
    B = args.batch_size
    T = args.sequence_length
    assert args.tokens_vld % (B * T * ddp_world_size) == 0
    val_steps = args.tokens_vld // (B * T * ddp_world_size)
    
    # --- Prepare data:
    loader_trn = DistributedDataLoader(args, ddp_rank, ddp_world_size)
    loader_vld = DistributedDataLoader(args, ddp_rank, ddp_world_size, vld=True)
    
    x, y = loader_trn.next_batch()

    # --- Init the model:
    model = Model(SimpleNamespace(
        vocab_size=50304, block_size=args.block_size,
        num_blocks=args.num_blocks, n_head=args.num_head, n_embd=768*2))

    if args.load_digital:
        model.load_state_dict(
            torch.load(args.load_digital + '/model.pth', map_location='cpu'))

    assert args.bb_num <= len(model.transformer.h)
    for num in range(args.bb_num):
        if args.replace_feedforward:
            d_inp = model.transformer.h[-1-num].mlp.c_fc.in_features
            d_out = model.transformer.h[-1-num].mlp.c_proj.out_features
            model.transformer.h[-1-num].mlp = ast.build(
                model.transformer.h[-1-num].mlp, d_inp, d_out)
        else:
            model.transformer.h[-1-num].mlp.c_fc = ast.build(
                model.transformer.h[-1-num].mlp.c_fc)

    model = model.cuda()
    model.master_process = master_process
    
    # --- Special trick for speeding up:
    if hasattr(torch_config, 'coordinate_descent_tuning'):
        torch_config.coordinate_descent_tuning = True # suggested by @Chillee

    model = torch.compile(model)
    
    # --- Wrap model into DDP container:
    if args.device_total > 1:
        model = DDP(model, device_ids=[gpu_num])
    else:
        model = DDP(model)
    model_raw = model.module # Always contains the unwrapped model
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float32)

    ast.prepare(model_raw)

    # --- Collect parameters for training proccess:
    params1 = model_raw.get_head_params()
    params2 = [] # model_raw.transformer.h.parameters()
    for param in model_raw.transformer.h.parameters():
        if not hasattr(param, 'ast_bb'):
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
    
    # --- Set the learning rate decay scheduler (linear warmup and warmdown):
    def get_lr(it):
        assert it <= args.epochs
        # 1) linear warmup for warmup_iters steps:
        if it < args.warmup_iters:
            return (it+1) / args.warmup_iters
        # 2) constant lr for a while:
        elif it < args.epochs - args.warmdown_iters:
            return 1.0
        # 3) linear warmdown:
        else:
            decay_ratio = (args.epochs - it) / args.warmdown_iters
            return decay_ratio
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr)
        for opt in optimizers]

    # --- Start the clock:
    torch.cuda.synchronize()
    t_start = tpc()

    # --- Training loop:
    loader_trn.reset()
    for step in range(args.epochs):
        print(f'DEBUG | Current step {step}; time: {tpc()-t_start:.2f}')

        last_step = (step == args.epochs-1)
        timed_steps = float('nan') if step <= 11 else (step - 10) + 1

        # --- Evaluate the validation dataset:
        do_vld = step == 0
        do_vld = do_vld or args.vld_every > 0 and (step+1) % args.vld_every == 0
        do_vld = do_vld or last_step
        if do_vld:
            t_start_vld = tpc()

            # Stop the clock:
            torch.cuda.synchronize()
            
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

            # Start the clock again:
            torch.cuda.synchronize()

            print(f'DEBUG | VLD time: {tpc()-t_start_vld:.2f}')

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
        ast.step()
        model.zero_grad(set_to_none=True)
        ast.step_before()

        if master_process and do_vld:
            l_trn = float(loss_trn.detach().cpu().numpy().item())
            l_vld = float(loss_vld.detach().cpu().numpy())
            ast.step_end(step, l_trn, l_vld, t=tpc()-t_start)

        if last_step:
            break

    if master_process:
        ast.done(model_raw)
        if ast.args.save_model:
            torch.save(model.state_dict(), ast.path('model_ddp.pth'))
    dist.destroy_process_group()


if __name__ == '__main__':
    run()