# Train all in one go
for red in 0.0 0.2 0.4 0.6 0.8 1.0
do
	#python3 reduce_train.py --opt BBB --red $red &
#	python3 reduce_train.py --opt VOGN  --red $red &
	#python3 reduce_train.py --opt HMC  --red $red &
#	python3 reduce_train.py --opt NA  --red $red &
#	python3 reduce_train.py --opt SWAG  --red $red
	python3 reduce_train.py --opt SGD  --red $red &
done
