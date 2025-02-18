# Defining Variables
agents=2
samples=1
max_iters=200
chk_freq=20
proj=ww_3
key=913528a8e92bf601b6eb055a459bcc89130c7f5f

result=$(python train.py --num-samples=$samples --num-agents=$agents \
    --checkpoint-freq=$chk_freq  --wandb-key=$key --wandb-project=$proj \
    --num-env-runners=30 | tail -n $samples)
    # For Testing:
    #--checkpoint-freq=2 --stop-iters=10 \

# Convert the results into an array
IFS=$'\n' read -rd '' -a results <<< $result

# Iterate through the output dir paths (is one per sample)
for path in "${results[@]}"; do
    # grep an array of the checkpoints
    chkpts=($(ls $path | grep "checkpoint"))

    # iterate through checkpoints
    for c in "${!chkpts[@]}"; do

        # Select a, number of agents to train
        for a in {3..8}; do
            # reduce the available training time left
            pretr_len=$((chk_freq*(c+1))) # Shift index by 1 
            re_iters=$((max_iters-pretr_len)) 
            python retrain.py --path=$path/${chkpts[c]} --num-samples=$samples \
            --num-env-runners=30 --num-agents=$a --steps_pretrained=$pretr_len \
            --coop=3 \
            --stop-iters=$re_iters --wandb-key=$key --wandb-project=$proj
            #--stop-iters=4
        done
    done
done

# tmux new-session -d 'bash ww_exp.sh'