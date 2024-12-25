# Defining Variables
agents=2
samples=1
max_iters=200
chk_freq=20

# Short run (for testing)
#result=$(python train.py --num-samples=$samples --num-env-runners=30 --num-agents=2 --stop-iters=10 --checkpoint-freq=2 | tail -n $samples)
# Longer:
result=$(python train.py --num-samples=$samples --num-env-runners=30 --num-agents=$agents --checkpoint-freq=$chk_freq --wandb-key=913528a8e92bf601b6eb055a459bcc89130c7f5f --wandb-project=delete_me | tail -n $samples) 

printf "\nPrinting result...\n"

# Convert the results into an array
IFS=$'\n' read -rd '' -a results <<< $result


# Iterate through the output dir paths (is one per sample)
for path in "${results[@]}"; do
    # grep an array of the checkpoints
    chkpts=($(ls $path | grep "checkpoint"))

    echo $chkpts

    # iterate through checkpoints
    for c in "${!chkpts[@]}"; do

        echo $path/${chkpts[c]}

        # Select a, number of agents to train to
        for a in {4,6,8}; do
            # reduce the available training time left
            pre_train_len=$((chk_freq*(c+1)))
            re_iters=$((max_iters-pre_train_len)) # Shift index by 1
            # echo $re_iters
            #python retrain.py --path=$path/${chkpts[c]} --num-samples=$samples --num-env-runners=30 --num-agents=$a --stop-iters=4 --pre-training=$pre_train_len
            python retrain.py --path=$path/${chkpts[c]} --num-samples=$samples --num-env-runners=30 --num-agents=$a --stop-iters=$re_iters --steps_pretrained=$pre_train_len --wandb-key=913528a8e92bf601b6eb055a459bcc89130c7f5f --wandb-project=delete_me
        done
    done
done

# --wandb-key=913528a8e92bf601b6eb055a459bcc89130c7f5f --wandb-project=delete_me
# tmux new-session -d 'bash exp.sh'