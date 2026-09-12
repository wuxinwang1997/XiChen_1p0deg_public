#!/bin/bash
#SBATCH -p 5090
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm-jupyter-%j.log
#SBATCH --error=slurm-jupyter-%j.log

# 非交互 shell 下激活 conda（登录 shell 的环境变量不随 sbatch 传递，必须重新 init + activate）
source /fs6/home/yangjh15/miniconda3/etc/profile.d/conda.sh
conda activate xichen1p0

# notebooks 用仓库相对路径（./data、./ckpts），需在此 cwd 下启动
cd /fs6/home/yangjh15/xichen/code/XiChen_1p0deg_public

JUPYTER_PORT=8890
# 该 echo 的 stdout 已由 #SBATCH --output 落入 slurm-jupyter-<JOBID>.log，用于查 node/端口
echo "Jupyter node: $(hostname)   port: ${JUPYTER_PORT}"
# --ip=0.0.0.0 绑定计算节点所有接口；--token=''/--password='' 免登录便于内网测试，
# 如需安全：去掉这两行并设置 --NotebookApp.token=真实token（URL 会打印带 token 的连接）。
jupyter notebook --no-browser --ip=0.0.0.0 --port=${JUPYTER_PORT} \
    --NotebookApp.token='' --NotebookApp.password='' \
    --NotebookApp.allow_root=True