# mode=CONLL2003 # CONLL2003 GUM I2B2 WNUT17
mode=inter # inter intra
N=5 # 1 5
K=1 # 1 5
num=20000
v=60
v2=60
topK=2

python llm4fewnerd.py --mode $mode \
         --N $N \
         --K $K \
         --topK $topK \
         --num $num \
         --v $v \
         --v2 $v2
