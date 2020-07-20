#python 3.7
from glob import glob
import librosa
import utils
import madmom

def beat_estimate(DB,GENRE,fout,name="***** Q1 *****\n",multi=1,method="own"):
    genres_p, genres_ALOTC = list(), list()
    for genre in GENRE:
        print('GENRE:', genre)
        FILES = glob(DB + '/wav/' + genre + '/*.wav')
        label, p_score, ALOTC_score = list(), list(), list()

        for f in FILES:
            f = f.replace('\\', '/')
            #print('FILE:', f)

            # Read the labeled tempo
            bpm = float(utils.read_tempofile(DB, genre, f))
            #print('ground-truth tempo: ', bpm)
            label.append(bpm)
            
            if method=="own":
                # Estimate a static tempo
                sr, y = utils.read_wav(f)
                hop_length = 512
                oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
                # predict the tempo1(slower one), tempo2(faster one)
                tempo1, tempo2 = utils.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length)
            elif method=="madmom":
                # madmom estimate tempo
                proc = madmom.features.tempo.TempoEstimationProcessor(fps=100)
                act = madmom.features.beats.RNNBeatProcessor()(f)
                tempo1 = (proc(act)).astype(float)[0][0].item()
                tempo2 = (proc(act)).astype(float)[1][0].item()
            # Q2
            tempo1 = tempo1*multi
            tempo2 = tempo2*multi

            # p score
            s1 = tempo1/(tempo1+tempo2)
            s2 = 1.0 - s1
            #print(s1, s2)
            p = s1 * utils.P_score(tempo1, bpm) + s2 * utils.P_score(tempo2, bpm)
            p_score.append(p)

            # ALOTC score
            ALOTC = utils.ALOTC(tempo1, tempo2, bpm)
            ALOTC_score.append(ALOTC)

            #print(p, ALOTC)

        p_avg = sum(p_score)/len(p_score)
        ALOTC_avg = sum(ALOTC_score)/len(ALOTC_score)
        genres_p.append(p_avg)
        genres_ALOTC.append(ALOTC_avg)
        print('finished : ',genre)

    print(genres_p)
    print(genres_ALOTC)
    fout.write(name)
    fout.write("Genre          \tP-score    \tALOTC score\n")
    for genre in range(len(GENRE)):
        fout.write("{:13s}\t{:8.2f}\t{:8.2f}\n".format(GENRE[genre], genres_p[genre], genres_ALOTC[genre]))
    fout.write('----------\n')
    fout.write("Overall P-score:\t{:.2f}\n".format(sum(genres_p)/len(genres_p)))
    fout.write("Overall ALOTC score:\t{:.2f}\n".format(sum(genres_ALOTC)/len(genres_ALOTC)))

if __name__ == '__main__':
    DB = 'data/Ballroom'
    GENRE = [g.split('/')[3] for g in glob(DB + '/wav/*')]

    #Q1
    fout = open('output/Q1.txt','a')
    beat_estimate(DB,GENRE,fout,name="***** Q1 *****\n",multi=1)
    fout.close()
    #Q2
    fout = open('output/Q2.txt','a')
    beat_estimate(DB,GENRE,fout,name="***** Q2_[T /2, T /2]*****\n",multi=1/2)
    beat_estimate(DB,GENRE,fout,name="***** Q2_[T /3, T /3]*****\n",multi=1/3)
    beat_estimate(DB,GENRE,fout,name="***** Q2_[T *2, T *2]*****\n",multi=2)
    beat_estimate(DB,GENRE,fout,name="***** Q2_[T *3, T *2]*****\n",multi=3)
    fout.close()
    #Q3
    fout = open('output/Q3.txt','a')
    beat_estimate(DB,GENRE,fout,name="***** Q3 *****\n",method="madmom")
    fout.close()