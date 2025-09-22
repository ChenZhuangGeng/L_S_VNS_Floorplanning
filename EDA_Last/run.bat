@echo off
echo =======================================================
echo            ��ʼִ��Anaconda���������Python�ű�
echo =======================================================

REM �л���Ŀ�깤��Ŀ¼
cd /d D:\IDEA2020\Project\EDA_Last

REM ����Conda����
echo [INFO] ���ڼ��� Conda ����: dl_pytorch...
call conda activate dl_pytorch

REM �����һ�������Ƿ�ɹ�
if %errorlevel% neq 0 (
    echo [ERROR] Conda ���� 'dl_pytorch' ����ʧ��!
    goto :eof
)

echo [INFO] Conda ��������ɹ�!

REM ���е�һ��Python�ű�
echo [INFO] �������� Runner �ű� (start=1, end=301)...
python -m src.run.Runner --start 1 --end 301

REM ���еڶ���Python�ű�
echo [INFO] �������� RunnerForClassify �ű� (start=1, end=301)...
python -m src.run.RunnerForClassify --start 1 --end 301

REM ���е�����Python�ű�
echo [INFO] �������� RunnerForGCN �ű� (start=1, end=301)...
python -m src.run.RunnerForGCN --start 1 --end 301

REM ���е��ĸ�Python�ű�
echo [INFO] �������� RunnerForGCNwithClassify �ű� (start=1, end=301)...
python -m src.run.RunnerForGCNwithClassify --start 1 --end 301

echo =======================================================
echo                      �������������
echo =======================================================

pause