#! /bin/bash
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 3-0
#SBATCH -p general
#SBATCH --gres=gpu:a100:3
#SBATCH -q public
#SBATCH --job-name=SwinV1-21K-Base_ChestX-ray14.sh
#SBATCH --output=/scratch/mthaku12/Supervised/slurm_op/SwinV1-21K-Base_ChestX-ray14-%j.out
#SBATCH --error=/scratch/mthaku12/Supervised/slurm_op/SwinV1-21K-Base_ChestX-ray14-%j.err
#SBATCH --mem=80G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mthaku12@asu.edu

# Function to echo the current time
echo_time() {
	echo "Timestamp: [$(/bin/date '+%Y-%m-%d %H:%M:%S')]......................................................$1"
}

echo "===== mthaku12 ====="
echo ""
echo ""

echo_time "[1/4] Loading module mamba"
module load mamba/latest
echo_time "[+] Done"
echo ""

echo_time "[2/4] Activating swin virtual environment"
source activate sl
echo_time "[+] Done"
echo ""

echo_time "[3/4] Changing working directory"
cd /scratch/mthaku12/Supervised

echo_time "[+] Done"
echo ""

echo_time "[4/4] Initiating code execution"

python main_classification.py \
	--model swin_base \
	--init imagenet_21k \
	--num_class 14 \
	--normalization chestx-ray \
	--data_dir /scratch/hmudigon/datasets/ssl/ChestX-ray14/images \
	--train_list dataset/Xray14_train_official.txt \
	--val_list dataset/Xray14_val_official.txt \
	--test_list dataset/Xray14_test_official.txt \
	--batch_size 512 \
	--epochs 200 \
	--exp_name SwinV1-21K-Base_ChestX-ray14 \
	--lr 0.001 \
	--opt adamw \
	--weight-decay 0.05 \
	--trial 3 \
	--warmup-epochs 20 \
	--workers 32 \
	--print_freq 25

echo_time "[+] Done"
echo ""
echo ""

echo_time "[+] Execution completed successfully!"
echo ""
echo "===== mthaku12 ====="
