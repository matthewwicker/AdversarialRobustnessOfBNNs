# Statistcal Estimation Runner
for (( INNUM=101; INNUM<=125; INNUM++ ))
do
	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset boston1 --epsilon 0.025 &
	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset energy1 --epsilon 0.025 &
	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset yacht1 --epsilon 0.025 &
	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset concrete1 --epsilon 0.025 &
	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset powerplant1 --epsilon 0.025 &
	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset wine1 --epsilon 0.025 &
	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset kin8nm1 --epsilon 0.025 &
	python3 EstimateStatisticalProperties.py --imnum $INNUM --dataset naval1 --epsilon 0.025 &
done
