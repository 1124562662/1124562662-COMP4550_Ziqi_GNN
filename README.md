# tmp
1. Download the pytorch c++ libraray in 
       https://pytorch.org/get-started/locally/
But I do not know if the codes are compatible with the latest version. So maybe it is better to download from:

https://drive.google.com/file/d/1-2G8DTXEfDLrA3jtGFuXD13xAgsokTh3/view?usp=sharing

2. Put the pytorch c++ libraray, namely the folder 'libtorch' in this folder.
3. Change the line 22 in file '/src/search/CMakeLists.txt':change it from 

        set(CMAKE_PREFIX_PATH /home/ziqi/prost-alpha/libtorch)
into 

        set(CMAKE_PREFIX_PATH Path_to_libtorch)
5. Go to test_bed folder and run 'python3 run-ser.py --all'.
6. run python3 build.py 
7. to run/evaluate the network, run:
 
          python3 prost.py instance_name  "[PROST -s 1 -se [selection2]]"
For example,

          python3 prost.py elevators_inst_mdp__1  "[PROST -s 1 -se [selection2]]"
7. to run the random policy,
          python3 prost.py instance_name  "[PROST -s 1 -se [randomselection]]"
