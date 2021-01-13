#TH - Death
TH = [100000,99511,99473,99446,99424,99406,99390,99376,99363,99350,99338,99325,99312,99296,99276,99250,99213,99163,99097,99015,98921,98820,98716,98612,98509,98406,98303,98198,98091,97982,97870,97756,97639,97517,97388,97249,97100,96939,96765,96576,96369,96141,95887,95606,95295,94952,94575,94164,93720,93244,92736,92196,91621,91009,90358,89665,88929,88151,87329,86460,85538,84558,83514,82399,81206,79926,78552,77078,75501,73816,72019,70105,68070,65914,63637,61239,58718,56072,53303,50411,47390,44234,40946,37546,34072,30575,27104,23707,20435,17338,14464,11852,9526,7498,5769,4331,3166,2249,1549,1032,663,410,244,139,75,39,19,9,4,2,1]

#TF - Life
TF = [100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,100000.000000000000000,99987.943454803000000,99976.891621705800000,99966.844500708300000,99956.797379710800000,99946.750258713400000,99935.698425616100000,99924.646592518900000,99912.590047321900000,99899.528790025200000,99883.453396429200000,99863.359154434300000,99839.246064040300000,99810.109413147700000,99776.953913856000000,99741.788990364800000,99706.624066873600000,99672.463855482200000,99637.262739119000000,99602.061622755800000,99562.808587152700000,99520.536087272300000,99474.237635022500000,99422.906742310700000,99365.536921044500000,99302.128171224000000,99232.680492849200000,99157.193885920100000,99073.655374251900000,98981.058469752200000,98879.403172420900000,98766.676506073400000,98642.878470709600000,98507.002578237200000,98358.042340563700000,98195.997757689200000,98004.419419985400000,97798.724783713900000,97579.922155915400000,97348.011536589700000,97104.001232777500000,96846.882937438200000,96573.631729450100000,96282.230994732200000,95970.664119203400000,95638.931102863600000,95264.475758390500000,94869.779584486300000,94452.818498208100000,94010.556375141100000,93537.933007927700000,93029.888189210000000,92482.373753101900000,91837.624008223600000,91136.853353442800000,90372.931854771900000,89538.729578223200000,88625.079465812800000,87620.777335560000000,86517.674691479100000,85305.585913587900000,83972.288257907500000,82501.484732466400000,80663.431201173300000,78624.843847641900000,76364.941964904500000,73862.944845993400000,71093.915642547400000,68033.956541553500000,64662.286836044200000,60970.594243232300000,56968.230081253200000,52685.326375210200000,48171.756821825700000,43502.331966183500000,38775.760166380100000,33634.170688786900000,28741.863237795800000,24153.183658068700000,19508.781438646500000,15400.161585067600000,11857.408133916700000,8887.683951047370000,6469.500438899080000,4562.745548692110000,3110.116353598300000,2042.849341423850000,1289.315853633560000,779.319804862954000,449.827975601043000,246.402585361081000,127.499012192652000,63.033219510974200,28.651463414079200,12.893158536335600,5.730292682815840,1.432573170703960]

#lx - table
lx = TH

'''
x = age
i = interest rate
n = maturity
m = number of payements
a = amount

'''

#Term Insurance Single Premium
def TermInsurance(x,n,i,a):
    NA = 0
    for j in range(1,n+1): NA+=MNQX(x,1,j-1) * TechDF(j,i)
    return NA * a


#Annuity : from 0 - to M-1 --> M values
def AnnuityFromZeroToM(x,i,m):
    A=0
    for j in range(m): A+= NPX(x,j)*TechDF(j,i)
    return A

#Term Insurance Annual Premium 
def TermInsuranceAnnual(x,m,n,i,a):
    NA = 0
    for j in range(1,n+1): NA+=MNQX(x,1,j-1) * TechDF(j,i)
    return  f'{((NA / AnnuityFromZeroToM(x,i,m) )* a):.2f}' 



'''utils'''
def Lx(x):
    return lx[x]
    
def LxOffset(x, offset):
    if x == 0:
        return lx[0]
    return max(0,lx[int(x)] - offset)

#dx, ndx, qx
def Dx(x):
    if x+1 == len(lx):
        return lx[x]
    return lx[x]-lx[x+1]

def NDX(x,n):
    return Lx(x) - Lx(int(x+n))

def Qx(x):
    #lx = []
    #lx = readLxInputFile(lx)
    if Lx(x) == 0:
        return 1
    return Dx(x)/Lx(x)

#Ex
def Ex(x):
    if x+1 == len(lx):
        return 0
    return sum(lx[x+1:])/lx[x]
#npx 
def NPX(x,n):
    return Lx(int(x+n))/Lx(x)

#nqx 
def NQX(x,n):
    return (Lx(x) - Lx(x+n) ) / Lx(x)


#mnqx 
def MNQX(x,n,m):
    return NPX(x,m) * NQX(x+m,n)


#techDF - actualization factor
def TechDF(n,i):
    return 1 / ((1+i)**n)



