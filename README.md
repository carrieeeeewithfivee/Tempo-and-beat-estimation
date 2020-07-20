# Tempo estimation, beat/downbeat tracking, and meter recognition of audio and symbolic data

## Prerequisites:

```
Python : 3.7
Scipy : 1.4.1
Madmom : 0.16.1
```
## Datasets:
* [Ballroom](https://drive.google.com/open?id=1Gk81pTyo65FIkUdR3inlNVWm72cqeaII)
* [SMC](http://smc.inesctec.pt/research/data-2/)
* [JCS](https://drive.google.com/drive/folders/18OP9LU8YflZtkULOk7qLAZkdBY8cOQfn)
## Run code:
Put all datasets in data folder, or change file paths in Task1 and Task
```
python Task1.py #for task I questions
python Task2.py #for task 2 and task 3 experiments
```
## Task 1: Tempo Estimation

#### Q1:

Design an algorithm that estimate the tempo of each clip in the Ballroom dataset. Note that your algorithm should output two predominant tempi for each clip: T1 (the slower one) and T2 (the faster one). Compute the average P-scores and the ALOTC scores of the eight genres.
#### Results:

```
***** Q1 *****
Genre              P-score        ALOTC score
Samba        	    0.51	    0.72
Tango        	    1.00	    1.00
ChaChaCha    	    0.99	    0.99
VienneseWaltz	    0.11	    0.14
Waltz        	    0.05	    0.07
Jive         	    0.57	    0.65
Rumba        	    0.79	    0.82
Quickstep    	    0.00	    0.00
----------
Overall P-score:	0.50
Overall ALOTC score:	0.55
```
#### Q2:

Instead of using your estimated [T1 ,T2 ] in evaluation, try to use [T1 /2, T2 /2], [T1 /3, T2 /3], [2T1 , 2T2 ], and [3T1 , 3T2 ] for estimation.

#### Results:

```
***** Q2_[T /2, T /2]*****
Genre              P-score    	ALOTC score
Samba        	    0.03	    0.03
Tango        	    0.00	    0.00
ChaChaCha    	    0.00	    0.00
VienneseWaltz	    0.00	    0.00
Waltz        	    0.13	    0.15
Jive         	    0.00	    0.00
Rumba        	    0.11	    0.12
Quickstep    	    0.00	    0.00
----------
Overall P-score:	0.03
Overall ALOTC score:	0.04

***** Q2_[T /3, T /3]*****
Genre          	P-score    	ALOTC score
Samba        	    0.00	    0.00
Tango        	    0.00	    0.00
ChaChaCha    	    0.00	    0.00
VienneseWaltz	    0.00	    0.00
Waltz        	    0.00	    0.00
Jive         	    0.00	    0.00
Rumba        	    0.00	    0.00
Quickstep    	    0.00	    0.00
----------
Overall P-score:	0.00
Overall ALOTC score:	0.00

***** Q2_[T *2, T *2]*****
Genre          	P-score    	ALOTC score
Samba        	    0.00	    0.00
Tango        	    0.00	    0.00
ChaChaCha    	    0.00	    0.00
VienneseWaltz	    0.11	    0.14
Waltz        	    0.00	    0.00
Jive         	    0.22	    0.43
Rumba        	    0.00	    0.00
Quickstep    	    0.92	    0.93
----------
Overall P-score:	0.16
Overall ALOTC score:	0.19

***** Q2_[T *3, T *2]*****
Genre          	P-score    	ALOTC score
Samba        	    0.00	    0.00
Tango        	    0.00	    0.00
ChaChaCha    	    0.00	    0.00
VienneseWaltz	    0.00	    0.00
Waltz        	    0.00	    0.00
Jive         	    0.00	    0.00
Rumba        	    0.00	    0.00
Quickstep    	    0.00	    0.00
----------
Overall P-score:	0.00
Overall ALOTC score:	0.00

```
#### Discussion:

Because tempo is heavily influenced by the harmonic, the same methods might have vastly different results on different genres of music. Quickstep has a P-score and ALOTC score of zero in Q1, but with T1 *2, T2 *2 it has P-score : 0.92 and ALOTC score 0.93. This shows that Quickstep has a faster tempo, twice as mush as estimated. This shows tweaking with calculations parameters can sometimes account for different harmonics.

#### Q3_1:
Using madmom to estimate tempo on the Ballroom dataset

#### Results:

```
***** Q3 *****
Genre          	P-score    	ALOTC score
Samba        	    0.56	    0.97
Tango        	    0.67	    1.00
ChaChaCha    	    0.65	    0.99
VienneseWaltz	    0.70	    1.00
Waltz        	    0.53	    0.96
Jive         	    0.67	    1.00
Rumba        	    0.51	    0.95
Quickstep    	    0.52	    0.78
----------
Overall P-score:	0.60
Overall ALOTC score:	0.96
```
#### Discussion:

I use madmom.features.tempo.TempoEstimationProcessor and beats.RNNBeatProcessor to achieve this. Data is fed through the RNN model, and the TempoEstimationProcessor takes the two most probable estimates as T1 and T2. It is evident that the results are better than that of Q1, especially for genres like VienneseWaltz. But others are not as accurate, like ChaChaCha which got a lower P-score.

## Task 2: using dynamic programming for beat tracking

#### Q3_2:

Using librosa.beat.beat_track to find the beat positions of a song

#### Results:

```
***** Q3_Ballroom *****
Genre          	F-score
Samba        	    0.54
Tango        	    0.74
ChaChaCha    	    0.82
VienneseWaltz	    0.71
Waltz        	    0.55
Jive         	    0.62
Rumba        	    0.73
Quickstep    	    0.55
----------
Overall F-score:	0.66
```
#### Q4:

Also use this algorithm on the SMC dataset and the JCS dataset. Compare the results to the Ballroom dataset.

#### Results:

```
***** Q4_SMC *****
Overall F-score:	0.36

***** Q4_JCS *****
Overall F-score:	0.64
```
#### Discussion, explain the difference in performance:

```
Ballroom achieves on average better results than JCS and SMC. I think this is due to the nature of the overall genre of the music in the dataset. Ballroom is mostly composed of dancing songs from BallroomDancers.com, with clear rhythms and unchanging beats. This makes beat tracking easier. JCS has time-varying meters, and SMC is mostly classical music, romantic music, film soundtracks... which focus more on melody than beats. This makes these two datasets harder to predict beats correctly.
```
#### Q5:
Use any function in madmom.features.beats for beat tracking and downbeat tracking in the Ballroom and the JCS dataset, and for beat tracking for the SMC dataset. For downbeat tracking, also compute the same F-score using tolerance of ±70 ms.
#### Results:

```
***** Q5_JCS_downbeat tracking with RNNDownBeatProcessor*****
Overall F-score:	0.76
***** Q5_Ballroom_downbeat tracking with RNNDownBeatProcessor*****
Genre          	F-score
Samba        	    0.91
Tango        	    0.84
ChaChaCha    	    0.91
VienneseWaltz	    0.98
Waltz        	    0.92
Jive         	    0.92
Rumba        	    0.91
Quickstep    	    0.88
----------
Overall F-score:	0.91
***** Q5_Ballroom_beat tracking with RNNBeatProcessor*****
Genre          	F-score
Samba        	    0.85
Tango        	    0.87
ChaChaCha    	    0.87
VienneseWaltz	    0.85
Waltz        	    0.82
Jive         	    0.83
Rumba        	    0.83
Quickstep    	    0.68
----------
Overall F-score:	0.83
***** Q5_SMC_beat tracking with RNNBeatProcessor *****
Overall F-score:	0.52
***** Q5_JCS_beat tracking with RNNBeatProcessor *****
Overall F-score:	0.83
```
#### Discussion. Compare the results to Q3 and Q4. How much improvement it gains?
I use madmom.features.downbeats.DBNDownBeatTrackingProcessor and RNNDownBeatProcessor for downbeats ; DBNDownBeatTrackingProcessor and RNNBeatProcessor for beat detection. The results are better than those of Q3 and Q4. They all rose 15%~20%.

### Task 3: meter recognition

#### Q6:
Although madmom.features.beats is the state-of-the-art downbeat tracker, one issue is that it assumes the meter of a song is constant. Actually, it assumes the meter a song is only one of the followings: 2-beats, 3-beats, 4-beats, 5-beats, 6-beats, and 7-beats, and the type of meter should be given by the parameter beats_per_bar. You may find some clips in the JCS dataset have time-varying meters, and madmom might perform not well in these clips. Could you design an algorithm to detect the instantaneous meter of a song?

#### Ans:

The DBNDownBeatTrackingProcessor uses one type of meter across the whole song. It finds the most probable meter in the process, so my hypothesis to detect the instantaneous meter is to cut the .wav data into chunks. This way the meter can change throughout the .wav data, though not within individual
chunks. I cannot calculate the frame-wise accuracy this way, so I did a series of experiments to verify my hypothesis. Looking at only one wav file (001_beats), I did the following experiments.


(001_beats, meter frame-wise accuracy is calculated by hand)
| Preprocess                     | F-score           | meter frame-wise accuracy  |
| ----------------------------|:----------------:| --------------------------------:|
| Whole song (none)         | 0.510               | 65.15%                                 |
| Split into 3 seconds       | 0.634               |  78.99%                                |
| Split into 2 seconds       | 0.666               |    81.07%                              |
| Split into 1 seconds       | 0.572               |    71.81%                              |


From the results in 001_beats, we can verify that this method works on this specific case. (The meter dynamically changes) So next I evaluated my method on the whole JCS dataset :


(JCS all)
| Preprocess        | F-score           |
| ------------- |:-------------:|
| Whole song (none)       | 0.760 | 
| Split into 3 seconds       | 0.756      | 
| Split into 2 seconds  | 0.762      |


This shows that though my hypothesis worked with 001_beats, it must backfire on other cases, for the overall F-score is mostly the same. So to further verify whether instantaneous meter benefits the performance using madmom beat and downbeat tracker, I designed another experiment on 001_beats.


(001_beats)
| Preprocess        | F-score           | meter frame-wise accuracy  |
| ------------- |:-------------:| -----:|
| Whole song (none)       | 0.510 | 65.15% |
| Give correct beats_per_bar       | 0.652      |  100% |


Note that “Give correct beats_per_bar” meter is given by me every time the meter changes, thus forcing the frame-wise accuracy to always be 100%.  
The meter changes in 001_beats are as follows:
```
Meter Changes (frame) = [4307,5562 ,5777 ,6218 ,6337 ,6506 ,6953 ,7086 ,7338 ,7400 ,7909 ,7974, 9275, 9339,10642,10875,11137,11367,11632,11861,12121,22501,23923,24635,27022]
Corresponding beats_per_bar = [ 4, 3, 4, 5, 2, 6, 5, 2, 4, 2, 4, 2, 4,2, 4, 7, 2, 7, 2, 7, 2, 4, 4, 4, 3]
```
This proves that for the 001_beats case, instantaneous meters helps the performance of downbeat tracking
using madmom beat and downbeat tracker.

