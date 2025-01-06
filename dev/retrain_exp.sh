arr=(a b c)

#exit 0

c=10

for n in "${!arr[@]}"; do
    for i in {2..4}; do
        #pretr_len=$((chk_freq*(c+1))) # Shift index by 1 
        #re_iters=$((max_iters-pre_train_len)) 
        a=$((n*(i+1)))

        echo $a

        b=$((a-c))

        echo $b

        echo "   "

        #echo "retraining $n agent for $i game..."
        #python retrain.py --num-env-runners=10 --path="/root/test/waterworld/PPO/${n}_agent/" --num-agents=$i \
        #--wandb-project=retrain-waterworld --wandb-key=913528a8e92bf601b6eb055a459bcc89130c7f5f
        #echo "done with $i game..."
    done
    #echo "done with $n agent..."
done
