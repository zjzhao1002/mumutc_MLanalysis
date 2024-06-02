import tensorflow as tf
from tensorflow import keras

import ROOT
import os

TMVA = ROOT.TMVA
TFile = ROOT.TFile

TMVA.Tools.Instance()

useBDT = True
useKeras = True
useDL = True

if ROOT.gSystem.GetFromPipe("root-config --has-tmva-pymva") == "yes":
    TMVA.PyMethodBase.PyInitialize()
else:
    useKeras = False

outputFile = TFile.Open("ClassificationOutput.root", "RECREATE")

factory = TMVA.Factory("mumutc_DNN_Classification", outputFile, V=False, ROC=True, Silent=False, Color=True, AnalysisType="Classification")

sigFileName = "mumutc_train.root"
bkgFileName1 = "mumubb_train.root"
bkgFileName2 = "mumucc_train.root"
bkgFileName3 = "mumuqq_train.root"
bkgFileName4 = "mumutt_train.root"
bkgFileName5 = "mumuwjj_train.root"
bkgFileName6 = "mumuww_train.root"
bkgFileName7 = "mumuzz_train.root"

sigFile = TFile.Open(sigFileName)
bkgFile1 = TFile.Open(bkgFileName1)
bkgFile2 = TFile.Open(bkgFileName2)
bkgFile3 = TFile.Open(bkgFileName3)
bkgFile4 = TFile.Open(bkgFileName4)
bkgFile5 = TFile.Open(bkgFileName5)
bkgFile6 = TFile.Open(bkgFileName6)
bkgFile7 = TFile.Open(bkgFileName7)

sigTree = sigFile.Get("eetc")
bkgTree1 = bkgFile1.Get("eetc")
bkgTree2 = bkgFile2.Get("eetc")
bkgTree3 = bkgFile3.Get("eetc")
bkgTree4 = bkgFile4.Get("eetc")
bkgTree5 = bkgFile5.Get("eetc")
bkgTree6 = bkgFile6.Get("eetc")
bkgTree7 = bkgFile7.Get("eetc")

# sigTree.Print()

loader = TMVA.DataLoader("dataset")
loader.AddVariable("njets")
loader.AddVariable("ptj1")
loader.AddVariable("ptj2")
loader.AddVariable("enerj1")
loader.AddVariable("enerj2")
loader.AddVariable("mjj")
loader.AddVariable("mj1")
loader.AddVariable("mj2")
loader.AddVariable("hb")
loader.AddVariable("lb")

totbkg = 7.15e3 + 3.56e4 + 1.73e4 + 9137.0 + 2.51e5 + 3.55e5 + 1.52e4
# print(totbkg)

sigWeight = 1.0 
bkgWeight1 = 9137.0 / totbkg
bkgWeight2 = 1.73e4 / totbkg
bkgWeight3 = 3.56e4 / totbkg
bkgWeight4 = 7.15e3 / totbkg
bkgWeight5 = 3.55e5 / totbkg
bkgWeight6 = 2.51e5 / totbkg
bkgWeight7 = 1.52e4 / totbkg

loader.AddSignalTree(sigTree, sigWeight)
loader.AddBackgroundTree(bkgTree1, bkgWeight1)
loader.AddBackgroundTree(bkgTree2, bkgWeight2)
loader.AddBackgroundTree(bkgTree3, bkgWeight3)
loader.AddBackgroundTree(bkgTree4, bkgWeight4)
loader.AddBackgroundTree(bkgTree5, bkgWeight5)
loader.AddBackgroundTree(bkgTree6, bkgWeight6)
loader.AddBackgroundTree(bkgTree7, bkgWeight7)

cuts = ROOT.TCut("")
cutb = ROOT.TCut("")

loader.PrepareTrainingAndTestTree(
    cuts, cutb, nTrain_Signal=40000, nTrain_Background=280000, SplitMode="Random", NormMode="NumEvents", V=False
)

if useBDT:
    factory.BookMethod(
        loader,
        TMVA.Types.kBDT,
        "BDT",
        V=False,
        NTrees=200,
        MinNodeSize="2.5%",
        MaxDepth=2,
        BoostType="AdaBoost",
        AdaBoostBeta=0.5,
        UseBaggedBoost=True,
        BaggedSampleFraction=0.5,
        SeparationType="GiniIndex",
        nCuts=20,
    )

if useDL:
    training1 = ROOT.TString(
        "LearningRate=1e-3,Momentum=0.9,"
        "ConvergenceSteps=10,BatchSize=128,TestRepetitions=1,"
        "MaxEpochs=20,WeightDecay=1e-4,Regularization=None,"
        "Optimizer=ADAM,ADAM_beta1=0.9,ADAM_beta2=0.999,ADAM_eps=1.E-7,"  # ADAM default parameters
        "DropConfig=0.0+0.0+0.0+0."
    )

    dnnMethodName = ROOT.TString("DNN_CPU")
    arch = "CPU"

    factory.BookMethod(
        loader,
        TMVA.Types.kDL,
        dnnMethodName,
        H=False,
        V=True,
        ErrorStrategy="CROSSENTROPY",
        VarTransform="G",
        WeightInitialization="XAVIER",
        InputLayout="1|1|10",
        BatchLayout="1|128|10",
        Layout="DENSE|64|TANH,DENSE|64|TANH,DENSE|64|TANH,DENSE|64|TANH,DENSE|1|LINEAR",
        TrainingStrategy=training1,
        Architecture=arch,
    )

useKeras = False
if useKeras:
    from tensorflow.python.keras import backend as K
    sess = tf.compat.v1.Session()
    K.set_session(sess)
    
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation='relu', input_dim=10))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', weighted_metrics=['accuracy'])
    model.save("model_mumutc.h5")
    model.summary()

    if not os.path.exists("model_mumutc.h5"):
        raise FileNotFoundError("Error in creating Keras model file")
    else:
        ROOT.Info("mumutc_DNN_Classification", "Booking Deep Learning Keras Model")
        factory.BookMethod(
            loader,
            TMVA.Types.kPyKeras,
            "PyKeras",
            H=True,
            V=False,
            VarTransform=None,
            FilenameModel="model_mumutc.h5",
            FilenameTrainedModel="trained_model_mumutc.h5",
            NumEpochs=20,
            BatchSize=100,
        )

factory.TrainAllMethods()

factory.TestAllMethods()

factory.EvaluateAllMethods()

c1 = factory.GetROCCurve(loader)
c1.Draw()

outputFile.Close()
