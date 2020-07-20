#python 3.7
from glob import glob
import librosa
import utils
import madmom
import mir_eval
import numpy as np

def beat_tracking(DB,fout,GENRE=None,dataset_name="Ballroom",name="***** Q4 *****\n",LB=None,method="librosa"):
    if dataset_name=="Ballroom":
        genres_F = list()
        for genre in GENRE:
            print('GENRE:', genre)
            FILES = glob(DB + '/wav/' + genre + '/*.wav')
            sum_f = 0.0
            cnt_f = 0.0

            for f in FILES:
                f = f.replace('\\', '/')
                print('FILE:', f)
                
                # Read the labeled tempo
                if method!="downbeat":
                    g_beats = utils.read_beatfile(DB,f, genre=genre, Dataset_name=dataset_name, LB=LB)
                else:
                    beats = utils.read_downbeatfile(DB,f,genre=genre,Dataset_name=dataset_name,LB=LB)
                    g_beats = []
                    for i in range(len(beats[1])):
                        if int(beats[1][i])==1: #is downbeat
                            g_beats.append(beats[0][i])
                # Beat tracking
                if method=="librosa":
                    sr, y = utils.read_wav(f)
                    _, beats = librosa.beat.beat_track(y=y, sr=sr)
                    timetag = librosa.frames_to_time(beats, sr=sr)
                    #print('detect beats:\n', timetag)
                elif method=="madmom":
                    proc = madmom.features.beats.BeatTrackingProcessor(fps=100)
                    act = madmom.features.beats.RNNBeatProcessor()(f)
                    timetag = proc(act)
                elif method=="downbeat":
                    proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4],fps=100)
                    act = madmom.features.downbeats.RNNDownBeatProcessor()(f)
                    timetags = proc(act)
                    timetag = []
                    for tag in timetags:
                        if int(tag[1])==1:
                            timetag.append(tag[0])
                
                    g_beats = np.array(g_beats)
                    timetag = np.array(timetag)

                # F score
                f_measure = mir_eval.beat.f_measure(g_beats, timetag, 0.07)
                sum_f += f_measure
                cnt_f += 1.0
            genres_F.append(sum_f/cnt_f)
            print('finished : ',genre)

        fout.write(name)
        fout.write("Genre          \tF-score\n")
        for g in range(len(GENRE)):
            fout.write("{:13s}\t{:8.2f}\n".format(GENRE[g], genres_F[g]))
        fout.write('----------\n')
        fout.write("Overall F-score:\t{:.2f}\n".format(sum(genres_F)/len(genres_F)))
    else: #the other datasets
        FILES = glob(DB + '/*.wav')
        FILES = sorted(FILES)
        sum_f = 0.0
        cnt_f = 0.0
        file_num = 0
        for f in FILES:
            print(f)
            f = f.replace('\\', '/')
            #print('FILE:', f)
            # Read the labeled tempo
            if method!="downbeat":
                g_beats = utils.read_beatfile(DB,f, genre=None, Dataset_name=dataset_name, LB=LB)
            else:
                beats = utils.read_downbeatfile(DB,f,genre=None,Dataset_name=dataset_name,LB=LB)
                g_beats = []
                for i in range(len(beats[1])):
                    if int(beats[1][i])==1: #is downbeat
                        g_beats.append(beats[0][i])

            # Beat tracking
            if method=="librosa":
                sr, y = utils.read_wav(f)
                _, beats = librosa.beat.beat_track(y=y, sr=sr)
                timetag = librosa.frames_to_time(beats, sr=sr)
            elif method=="madmom":
                proc = madmom.features.beats.BeatTrackingProcessor(fps=100)
                act = madmom.features.beats.RNNBeatProcessor()(f)
                timetag = proc(act)
            elif method=="downbeat":
                proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[2,3,4,5,6,7],fps=100)
                act = madmom.features.downbeats.RNNDownBeatProcessor()(f)
                timetags = proc(act)
                timetag = []
                print("predicted: ",timetags)
                for tag in timetags:
                    if int(tag[1])==1:
                        timetag.append(tag[0])        
                g_beats = np.array(g_beats)
                timetag = np.array(timetag)
            #try dynamic tracking
            elif method=="mine":
                act = madmom.features.downbeats.RNNDownBeatProcessor()(f)       
                timetags = []
                
                #split into intervals
                cut_places = []
                beat_bar = []
                for i in range(300,len(act),300):
                    cut_places.append(i)
                    beat_bar.append(4) #fixed intervals

                '''#-----------------------------
                #split into custom intervals
                cut_places = [4307,5562 ,5777 ,6218 ,6337 ,6506 ,6953 ,7086 ,7338 ,7400 ,7909 ,7974, 9275,
                              9339,10642,10875,11137,11367,11632,11861,12121,22501,23923,24635,27022]
                beat_bar   = [   4,   3,    4,    5,    2,     6,   5,    2,    4,    2,    4,    2,    4,
                                 2,   4,    7,    2,    7,     2,   7,    2,    4,    4,    4,    3]
                #-----------------------------'''

                begin_cut = 0
                act_list = []
                for cut in cut_places:
                    try:
                        act_list.append(act[begin_cut:cut,:])
                        begin_cut = cut
                    except Exception as e:
                        print("cut error: ",e)

                add_time = 0
                for a,b,c in zip(act_list,beat_bar,cut_places):
                    try:
                        proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[2,3,4,5,6,7],fps=100)
                        t = proc(a)
                        '''print("~~~~~~~~~~~~")
                        print("act: ",a)
                        print("assigned beat: ",b)
                        print("predicted: ",t)
                        print("time margin: ", add_time, " ~ ", t[-1,0]+add_time)
                        print("~~~~~~~~~~~~")'''
                        for tag in t:
                            if int(tag[1])==1:
                                timetags.append(tag[0] + add_time)
                        add_time = c/100
                    except Exception as e:
                        print("run madmom error: ",e)
                g_beats = np.array(g_beats)
                timetag = np.array(timetags)
            # F score
            f_measure = mir_eval.beat.f_measure(g_beats, timetag, 0.07)
            sum_f += f_measure
            cnt_f += 1.0
            file_num = file_num+1
            #if file_num%10 == 0:
            #    print("Finished " +str(file_num)+" files !")
        
        print("Overall F-score:",sum_f/cnt_f)
        fout.write(name)
        fout.write("Overall F-score:\t{:.2f}\n".format(sum_f/cnt_f))
        
        

if __name__ == '__main__':
    DB = 'data/Ballroom'
    GENRE = [g.split('/')[3] for g in glob(DB + '/wav/*')]
    JCS_DB = 'data/JCS_dataset/audio'
    JCS_LB = 'data/JCS_dataset/annotations'
    SMC_DB = 'data/SMC_MIREX/SMC_MIREX_Audio'
    SMC_LB = 'data/SMC_MIREX/SMC_MIREX_Annotations'
    #Q3,4
    fout = open('output/Q4.txt','a')
    beat_tracking(DB,fout,GENRE=GENRE,dataset_name="Ballroom",name="***** Q4_Ballroom *****\n")
    beat_tracking(SMC_DB,fout,LB=SMC_LB,dataset_name="SMC",name="***** Q4_SMC *****\n")
    beat_tracking(JCS_DB,fout,LB=JCS_LB,dataset_name="JCS",name="***** Q4_JCS *****\n")
    fout.close()
    ##Q5
    fout = open('output/Q5.txt','a')
    beat_tracking(DB,fout,GENRE=GENRE,dataset_name="Ballroom",name="***** Q5_Ballroom_beat tracking with RNNBeatProcessor*****\n",method="madmom")
    beat_tracking(SMC_DB,fout,LB=SMC_LB,dataset_name="SMC",name="***** Q5_SMC_beat tracking with RNNBeatProcessor *****\n",method="madmom")
    beat_tracking(JCS_DB,fout,LB=JCS_LB,dataset_name="JCS",name="***** Q5_JCS_beat tracking with RNNBeatProcessor *****\n",method="madmom")
    beat_tracking(DB,fout,GENRE=GENRE,dataset_name="Ballroom",name="***** Q5_Ballroom_downbeat tracking with RNNDownBeatProcessor*****\n",method="downbeat")
    beat_tracking(JCS_DB,fout,LB=JCS_LB,dataset_name="JCS",name="***** Q5_JCS_downbeat tracking with RNNDownBeatProcessor*****\n",method="downbeat")
    fout.close()
    ##Q6
    fout = open('output/Q6.txt','a')
    #beat_tracking(JCS_DB,fout,LB=JCS_LB,dataset_name="JCS",name="***** Q6_JCS *****\n",method="downbeat")
    beat_tracking(JCS_DB,fout,LB=JCS_LB,dataset_name="JCS",name="***** Q6_JCS_3sec *****\n",method="mine")
    fout.close()