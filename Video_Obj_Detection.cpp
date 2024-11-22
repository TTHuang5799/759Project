#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=task1.output
#SBATCH --error=task1.err

cd $SLURM_SUBMIT_DIR

module load gcc/13.2.0
module load nvidia/cuda

if [ -d "repo759" ]; then
  cd repo759
  rm -rf .git
  cd ..
  rm -r repo759
fi

git clone https://github.com/TTHuang5799/repo759.git
echo "repository exists"
cd repo759/HW02


if [ "$1" = "1c" ]; then
    g++ scan.cpp task1.cpp -Wall -O3 -std=c++17 -o task1

    for i in {10..30}; do
        n=$((2**i))
        echo "Running task1 for value n = 2^$i"
        ./task1 $n  
        echo 
    done
else 
    g++ scan.cpp task1.cpp -Wall -O3 -std=c++17 -o task1
    echo "Running task1 for value n = $1"
    ./task1 $1
fi
