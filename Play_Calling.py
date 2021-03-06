import pandas as pd

data = pd.read_csv("Book3.csv")
blowout = data[data["Margin"] <= -21]
ahead = data[(data["Margin"] <= -8) & (data["Margin"]>= -28)]
close = data[(data["Margin"] >= -8) & (data["Margin"] <= 8)]

blowout = blowout[(blowout["PLAY TYPE"] == "Pass") | (blowout["PLAY TYPE"] == "Run")]
b = blowout["PLAY TYPE"].value_counts()
print("Blowout Pass Percentage: %.2f Run Percentage: %.2f " % (b["Pass"]/b.sum(), b["Run"]/b.sum()))

ahead = ahead[(ahead["PLAY TYPE"] == "Pass") | (ahead["PLAY TYPE"] == "Run")]
a = ahead["PLAY TYPE"].value_counts()
print("Ahead Pass Percentage: %.2f Run Percentage: %.2f " % (a["Pass"]/a.sum(), a["Run"]/a.sum()))

close = close[(close["PLAY TYPE"] == "Pass") | (close["PLAY TYPE"] == "Run")]
c = close["PLAY TYPE"].value_counts()
print("Close  Pass Percentage: %.2f Run Percentage: %.2f " % (c["Pass"]/c.sum(), c["Run"]/c.sum()))

data = data[(data["PLAY TYPE"] == "Pass") | (data["PLAY TYPE"] == "Run")]
d = data["PLAY TYPE"].value_counts()
print("All Down Pass Percentage: %.2f Run Percentage: %.2f " % (d["Pass"]/d.sum(), d["Run"]/d.sum()))

first_down = data[data["DN"] == 1]
fShort = first_down[first_down["DIST"] <=3]
fs = fShort["PLAY TYPE"].value_counts()
print("1st & Short Run Percentage: %.2f  " % (fs["Run"]/fs.sum()))
fMid = first_down[(first_down["DIST"] >3)  & (first_down["DIST"] <=7)]
fm=fMid["PLAY TYPE"].value_counts()
print("1st & Medium Pass Percentage: %.2f Run Percentage: %.2f " % (fm["Pass"]/fm.sum(), fm["Run"]/fm.sum()))
fLong = first_down[first_down["DIST"] >7]
fl=fLong["PLAY TYPE"].value_counts()
print("1st & Long Pass Percentage: %.2f Run Percentage: %.2f " % (fl["Pass"]/fl.sum(), fl["Run"]/fl.sum()))

second_down = data[data["DN"] == 2]
sShort = second_down[second_down["DIST"] <=3]
ss = sShort["PLAY TYPE"].value_counts()
print("2nd & Short Pass Percentage: %.2f Run Percentage: %.2f " % (ss["Pass"]/ss.sum(), ss["Run"]/ss.sum()))

sMid = second_down[(second_down["DIST"] >3)  & (second_down["DIST"] <=7)]
sm = sMid["PLAY TYPE"].value_counts()
print("2nd &  Medium Pass Percentage: %.2f Run Percentage: %.2f " % (sm["Pass"]/sm.sum(), sm["Run"]/sm.sum()))


sLong = second_down[second_down["DIST"] >7]
sl = sLong["PLAY TYPE"].value_counts()
print("2nd & Long Pass Percentage: %.2f Run Percentage: %.2f " % (sl["Pass"]/sl.sum(), sl["Run"]/sl.sum()))

third_down = data[data["DN"] == 3]
third_down["PLAY TYPE"].value_counts()
tShort = third_down[third_down["DIST"] <=3]
ts = tShort["PLAY TYPE"].value_counts()
print("3rd & Short Pass Percentage: %.2f Run Percentage: %.2f " % (ts["Pass"]/ts.sum(), ts["Run"]/ts.sum()))

tMid = third_down[(third_down["DIST"] >3)  & (third_down["DIST"] <=7)]
tm = tMid["PLAY TYPE"].value_counts()
tm = tMid["PLAY TYPE"].value_counts()
print("3rd & Medium Pass Percentage: %.2f Run Percentage: %.2f " % (tm["Pass"]/tm.sum(), tm["Run"]/tm.sum()))

tLong = third_down[third_down["DIST"] >7]
tl = tLong["PLAY TYPE"].value_counts()
print("3rd & Long Pass Percentage: %.2f Run Percentage: %.2f " % (tl["Pass"]/tl.sum(), tl["Run"]/tl.sum()))

heavy_pass=[]
heavy_run=[]
balanced=[]

for i in data["OFF FORM"].unique():
    data = data[(data["PLAY TYPE"] == "Pass") | (data["PLAY TYPE"] == "Run")]
    da = data[data["OFF FORM"] == i]
    d = da["PLAY TYPE"].value_counts().reindex(
    data["PLAY TYPE"].unique(), fill_value=0)
    if(d["Pass"] / d.sum() * 100 > 65):
        heavy_pass.append(i)
    elif(d["Run"] / d.sum() * 100 > 65):
        heavy_run.append(i)
    else:
        balanced.append(i)

    print(str(i)+" Pass Percentage: %.2f Run Percentage: %.2f " % (d["Pass"] / d.sum(), d["Run"] / d.sum()))
print(str(heavy_run) +"\n"+str(heavy_pass)+"\n"+str(balanced))