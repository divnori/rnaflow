for gpu in 0 1 2 3 4 5 6 7; do
	bash batch_fast.sh $gpu &
done
