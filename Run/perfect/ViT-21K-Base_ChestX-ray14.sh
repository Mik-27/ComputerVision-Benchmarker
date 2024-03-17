#! /bin/bash
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 3-0
#SBATCH -p general
#SBATCH --gres=gpu:a100:3
#SBATCH -q public
#SBATCH --job-name=ViT-21K-Base_ChestX-ray14
#SBATCH --output=/scratch/hmudigon/Acad/CSE598-ODL/Supervised/slurm_op/ViT-21K-Base_ChestX-ray14-%j.out
#SBATCH --error=/scratch/hmudigon/Acad/CSE598-ODL/Supervised/slurm_op/ViT-21K-Base_ChestX-ray14-%j.err
#SBATCH --mem=80G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hmudigon@asu.edu

# Function to echo the current time
echo_time() {
	echo "Timestamp: [$(/bin/date '+%Y-%m-%d %H:%M:%S')]......................................................$1"
}

echo "===== himudigonda ====="
echo ""
echo ""

echo_time "[1/4] Loading module mamba"
module load mamba/latest
echo_time "[+] Done"
echo ""

echo_time "[2/4] Activating ViT virtual environment"
source activate sl
echo_time "[+] Done"
echo ""

echo_time "[3/4] Changing working directory"
cd /scratch/hmudigon/Acad/CSE598-ODL/Supervised

echo_time "[+] Done"
echo ""

echo_time "[4/4] Initiating code execution"

python main_classification.py \
	--model vit_base \
	--init imagenet_21k \
	--num_class 14 \
	--normalization chestx-ray \
	--data_dir /scratch/hmudigon/datasets/ChestX-ray14/images \
	--train_list dataset/Xray14_train_official.txt \
	--val_list dataset/Xray14_val_official.txt \
	--test_list dataset/Xray14_test_official.txt \
	--batch_size 512 \
	--epochs 200 \
	--exp_name ViT-21K-Base_ChestX-ray14 \
	--lr 0.01 \
	--opt sgd \
	--trial 10 \
	--warmup-epochs 0 \
	--workers 32 \
	--print_freq 25
	
echo_time "[+] Done"
echo ""
echo ""

echo_time "[+] Execution completed successfully!"
echo ""
echo "===== himudigonda ====="
