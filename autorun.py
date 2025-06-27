import subprocess

def autorun_cnn_cifar():
    ranks = [1, 3, 5, 7, 10, 50, 100]
    samples_list = [1, 10, 50, 100, 500]

    for rank in ranks:
        for samples_val in samples_list:
            exp_name = f"bb_matvec_rank{rank}_samples{samples_val}"
            
            command = [
                "python", "script.py",
                "--task", "cnn_cifar",
                "--mode", "bb",
                "--name", exp_name,
                "--rank", str(rank),
                "--samples_bb", str(samples_val),
                "--samples_sm", str(samples_val)
            ]
            
            print(f"\nЗапуск: rank={rank}, samples={samples_val}")
            print("Команда:", " ".join(command))
            
            try:
                result = subprocess.run(command, check=True)
                print(f"Успешно запущено: {exp_name}", result)
            except subprocess.CalledProcessError as e:
                print(f"Ошибка при выполнении: {exp_name}")
                print(f"Код ошибки: {e.returncode}")

    print("\nВсе эксперименты завершены!")


if __name__ == '__main__':
    autorun_cnn_cifar()