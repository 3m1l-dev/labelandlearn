import pandas as pd
from sklearn.utils import shuffle

jan05 = pd.read_csv("jan05.csv")
jan06 = pd.read_csv("jan06.csv")
jan07 = pd.read_csv("jan07.csv")
jan08 = pd.read_csv("jan08.csv")
jan09 = pd.read_csv("jan09.csv")
jan10 = pd.read_csv("jan10.csv")
jan11 = pd.read_csv("jan11.csv")
jan12 = pd.read_csv("jan12.csv")
jan13 = pd.read_csv("jan13.csv")
jan14 = pd.read_csv("jan14.csv")
jan15 = pd.read_csv("jan15.csv")
jan16 = pd.read_csv("jan16.csv")
jan17 = pd.read_csv("jan17.csv")
jan18 = pd.read_csv("jan18.csv")
jan19 = pd.read_csv("jan19.csv")
jan20 = pd.read_csv("jan20.csv")
jan21 = pd.read_csv("jan21.csv")
jan22 = pd.read_csv("jan22.csv")
jan23 = pd.read_csv("jan23.csv")
jan24 = pd.read_csv("jan24.csv")
jan25 = pd.read_csv("jan25.csv")
jan26 = pd.read_csv("jan26.csv")
jan27 = pd.read_csv("jan27.csv")
jan28 = pd.read_csv("jan28.csv")
jan29 = pd.read_csv("jan29.csv")
jan30 = pd.read_csv("jan30.csv")
jan31 = pd.read_csv("jan31.csv")
feb01 = pd.read_csv("feb01.csv")
feb02 = pd.read_csv("feb02.csv")
feb03 = pd.read_csv("feb03.csv")
feb04 = pd.read_csv("feb04.csv")
feb05 = pd.read_csv("feb05.csv")
feb06 = pd.read_csv("feb06.csv")
feb07 = pd.read_csv("feb07.csv")
feb08 = pd.read_csv("feb08.csv")
feb09 = pd.read_csv("feb09.csv")
feb10 = pd.read_csv("feb10.csv")
feb11 = pd.read_csv("feb11.csv")
feb12 = pd.read_csv("feb12.csv")
feb13 = pd.read_csv("feb13.csv")
feb14 = pd.read_csv("feb14.csv")
feb15 = pd.read_csv("feb15.csv")
feb16 = pd.read_csv("feb16.csv")
feb17 = pd.read_csv("feb17.csv")
feb18 = pd.read_csv("feb18.csv")
feb19 = pd.read_csv("feb19.csv")
feb20 = pd.read_csv("feb20.csv")
feb21 = pd.read_csv("feb21.csv")
feb22 = pd.read_csv("feb22.csv")
feb23 = pd.read_csv("feb23.csv")
feb24 = pd.read_csv("feb24.csv")
feb25 = pd.read_csv("feb25.csv")
feb26 = pd.read_csv("feb26.csv")
feb27 = pd.read_csv("feb27.csv")
feb28 = pd.read_csv("feb28.csv")
mar01= pd.read_csv("mar01.csv")
mar02= pd.read_csv("mar02.csv")
mar03= pd.read_csv("mar03.csv")
mar04= pd.read_csv("mar04.csv")
mar05= pd.read_csv("mar05.csv")
mar06= pd.read_csv("mar06.csv")
mar07= pd.read_csv("mar07.csv")
mar08= pd.read_csv("mar08.csv")
mar09= pd.read_csv("mar09.csv")
mar10= pd.read_csv("mar10.csv")
mar11= pd.read_csv("mar11.csv")
mar12= pd.read_csv("mar12.csv")
mar13= pd.read_csv("mar13.csv")
mar14= pd.read_csv("mar14.csv")
mar15= pd.read_csv("mar15.csv")
mar16= pd.read_csv("mar16.csv")
mar17= pd.read_csv("mar17.csv")
mar18= pd.read_csv("mar18.csv")
mar19= pd.read_csv("mar19.csv")
mar20= pd.read_csv("mar20.csv")
mar21= pd.read_csv("mar21.csv")
mar22= pd.read_csv("mar22.csv")
mar23= pd.read_csv("mar23.csv")
mar24= pd.read_csv("mar24.csv")
mar25= pd.read_csv("mar25.csv")
mar26= pd.read_csv("mar26.csv")
mar27= pd.read_csv("mar27.csv")
mar28= pd.read_csv("mar28.csv")
mar29= pd.read_csv("mar29.csv")
mar30= pd.read_csv("mar30.csv")
mar31= pd.read_csv("mar31.csv")
apr01= pd.read_csv("apr01.csv")
apr02= pd.read_csv("apr02.csv")
apr03= pd.read_csv("apr03.csv")
apr04= pd.read_csv("apr04.csv")
apr05= pd.read_csv("apr05.csv")
apr06= pd.read_csv("apr06.csv")
apr07= pd.read_csv("apr07.csv")

frames = [
        jan05,
        jan06,
        jan07,
        jan08,
        jan09,
        jan10,
        jan11,
        jan12,
        jan13,
        jan14,
        jan15,
        jan16,
        jan17,
        jan18,
        jan19,
        jan20,
        jan21,
        jan22,
        jan23,
        jan24,
        jan25,
        jan26,
        jan27,
        jan28,
        jan29,
        jan30,
        jan31,
        feb01,
        feb02,
        feb03,
        feb04,
        feb05,
        feb06,
        feb07,
        feb08,
        feb09,
        feb10,
        feb11,
        feb12,
        feb13,
        feb14,
        feb15,
        feb16,
        feb17,
        feb18,
        feb19,
        feb20,
        feb21,
        feb22,
        feb23,
        feb24,
        feb25,
        feb26,
        feb27,
        feb28,
        mar01,
        mar02,
        mar03,
        mar04,
        mar05,
        mar06,
        mar07,
        mar08,
        mar09,
        mar10,
        mar11,
        mar12,
        mar13,
        mar14,
        mar15,
        mar16,
        mar17,
        mar18,
        mar19,
        mar20,
        mar21,
        mar22,
        mar23,
        mar24,
        mar25,
        mar26,
        mar27,
        mar28,
        mar29,
        mar30,
        mar31,
        apr01,
        apr02,
        apr03,
        apr04,
        apr05,
        apr06,
        apr07
        ]

dataset = pd.concat(frames)
dataset = shuffle(dataset)
dataset.to_csv("classifier_dataset.csv", encoding='utf-8', index=False)
