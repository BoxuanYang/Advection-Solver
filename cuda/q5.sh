GRID_DIM=$1
BLOCK_DIM=$2
WIDTH=$3
HEIGHT=$4

echo "Doing Experiments on Bx, By"
echo "-----------------------------"
for Bx in 1 2 4 8 16 32 64 1024; do
    # Calculate By as 1024 / Bx
    By=$((1024 / Bx))
    echo ""
    echo "Bx = $Bx, By = $By" 
    
    # Execute the ./testAdvect command with the calculated values
    ./testAdvect -g 32,32 -b ${Bx},${By} 4096 4096 10
done

echo "Doing Experiments on Gx, Gy"
echo "-----------------------------"
for Gx in 1 2 4 8 16 32 64 128 256; do
    # Calculate By as 1024 / Bx
    Gy=$((256 / Gx))
    echo ""
    echo "Gx = $Gx, Gy = $Gy" 
    
    # Execute the ./testAdvect command with the calculated values
    ./testAdvect -g ${Gx},${Gy} -b $32,32 4096 4096 10
done