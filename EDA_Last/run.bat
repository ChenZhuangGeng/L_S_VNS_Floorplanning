@echo off
echo =======================================================
echo            开始执行Anaconda环境激活和Python脚本
echo =======================================================

REM 切换到目标工作目录
cd /d D:\IDEA2020\Project\EDA_Last

REM 激活Conda环境
echo [INFO] 正在激活 Conda 环境: dl_pytorch...
call conda activate dl_pytorch

REM 检查上一条命令是否成功
if %errorlevel% neq 0 (
    echo [ERROR] Conda 环境 'dl_pytorch' 激活失败!
    goto :eof
)

echo [INFO] Conda 环境激活成功!

REM 运行第一个Python脚本
echo [INFO] 正在运行 Runner 脚本 (start=1, end=301)...
python -m src.run.Runner --start 1 --end 301

REM 运行第二个Python脚本
echo [INFO] 正在运行 RunnerForClassify 脚本 (start=1, end=301)...
python -m src.run.RunnerForClassify --start 1 --end 301

REM 运行第三个Python脚本
echo [INFO] 正在运行 RunnerForGCN 脚本 (start=1, end=301)...
python -m src.run.RunnerForGCN --start 1 --end 301

REM 运行第四个Python脚本
echo [INFO] 正在运行 RunnerForGCNwithClassify 脚本 (start=1, end=301)...
python -m src.run.RunnerForGCNwithClassify --start 1 --end 301

echo =======================================================
echo                      所有任务已完成
echo =======================================================

pause