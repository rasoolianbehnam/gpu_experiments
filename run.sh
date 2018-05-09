#clang++ test.cpp -o test && ./test
#clang++ potential.cpp -o potential && ./potential
#nvcc add2.cu -o add2_cu && ./add2_cu
#nvcc add3.cu -o add3_cu && ./add3_cu
#nvprof python $1
clear&&clear
fileName=$(echo $1 | cut -d'.' -f1)
echo file name: $fileName
if [[ $1 = *.cu ]]; 
then 
    nvcc $1 -o ${fileName}_cu && ./${fileName}_cu;
elif [[ $1 = *.py ]]; 
then
    python $1;
elif [[ $1 = *.c* ]]; 
then
    echo gcc -c $1 -o tmp.o | bash -x && echo gcc -static tmp.o -lm -o ${fileName}.out | bash -x && echo ./${fileName}.out | bash -x
    #echo gcc $1 -o ${fileName}_cpp | bash -x && echo ./${fileName}_cpp | bash -x
elif [[ $1 = *.m ]]; 
then
    octave $1;
 
fi
