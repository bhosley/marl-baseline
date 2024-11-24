for n in {2..8}; 
do
    for i in {2..8}; 
    do
        echo "retraining $n agent for $i game..."
        python retrain.py --num-env-runners=10 --path="/root/test/waterworld/PPO/${n}_agent/" --num-agents=$i \
        --wandb-project=retrain-waterworld --wandb-key=913528a8e92bf601b6eb055a459bcc89130c7f5f
        echo "done with $i game..."
    done
    echo "done with $n agent..."
done
