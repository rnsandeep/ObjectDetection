for i in $(seq 10000 5000 15000)
do
   ./experiments/scripts/test_faster_rcnn.sh  0 pascal_voc res101 $i
done

