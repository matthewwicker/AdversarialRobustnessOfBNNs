# Statistcal Estimation Runner
for (( INNUM=0; INNUM<=100; INNUM++ ))
do
	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset boston1 &
	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset energy1 &
	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset yacht1 &
	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset concrete1 &
	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset powerplant1 &
	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset wine1 &
	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset kin8nm1 &
	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset naval1

done
