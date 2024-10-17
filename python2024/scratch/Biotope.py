# def summary(palter, pstruktur, pbio1lbp, plrganteil):
#     if palter & pstruktur & pbio1lbp & plrganteil is not None:
#         summarystring = palter + pstruktur + pbio1lbp + plrganteil
#         return summarystring
#     else:
#         return "Incomplete"


def summary(palter, pstruktur, pbio1lbp, plrganteil):
    inputlist=[palter, pstruktur, pbio1lbp, plrganteil]
    summarystring=""
    print(inputlist)
    
    
    
        
    for i in range(len(inputlist)):
        
        print("i"+str(i)+inputlist[i]+"len"+str(len(inputlist[i].strip()))+"lentotal"+str(len(inputlist)))
       
        if len(inputlist[i].strip()) > 0 and i < len(inputlist) - 1:
            summarystring += str(inputlist[i]).strip() + ", "
        elif i == len(inputlist) - 1:
            summarystring += str(inputlist[i]).strip()
            break           
            
    return summarystring


print(summary("ta2"," ","BD3","100"))


 


