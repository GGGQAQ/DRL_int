import subprocess

# 定义要执行的脚本列表
script_list = ['1.py', '2.py']

# 循环执行每个脚本
for script in script_list:
    try:
        # 使用 subprocess 运行脚本
        subprocess.run(['python', script], check=True)
        print(f'Script {script} executed successfully.')
    except subprocess.CalledProcessError as e:
        # 如果脚本执行失败，捕获异常并输出错误信息
        print(f'Error executing script {script}: {e}')

print('All scripts executed.')
