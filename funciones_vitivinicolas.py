# FUNCIONES DEL PROGAMA
# @title carga de librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

años=["2013","2018","2022","2023"]

# @title gráfico matriz var
def graf_matriz_doble_filtro_var(dataframe,dataframe2,x,y):
  data11=matriz_doble_filtro_2(dataframe,x,y)
  data22=matriz_doble_filtro_2(dataframe2,x,y)
  filas11=list(data11.iloc[:,0].values)
  filas22=list(data22.iloc[:,0].values)
  columnas11=list(data11.columns.values)
  columnas22=list(data22.columns.values)
  if (x=="Variedad") or (y=="Variedad"):
    for i in range(0,len(data11.iloc[:,0].values)):
      if data11.iloc[i,0]== "SAUVIGNON":
        data11.iloc[i,0]== "SAUVIGNON_"
    for j in range(0,len(data22.iloc[:,0].values)):
      if data22.iloc[j,0]== "SAUVIGNON":
        data22.iloc[j,0]== "SAUVIGNON_"
  for i in range(0,len(columnas11)):
    for j in range(0,len(columnas22)):
      if columnas11[i].__contains__(columnas22[j]):
        columnas11[i]=columnas22[j]
  for s in range(0,len(data11.columns.values)):
    data11.columns.values[s]=columnas11[s]
  for i in range(0,len(filas11)):
    for j in range(0,len(filas22)):
      if filas11[i].__contains__(filas22[j]):
        filas11[i]=filas22[j]
  for k in range(0,len(data11.iloc[:,0].values)):
    data11.iloc[k,0]=filas11[k]
  for i in range(0,len(filas11)):
    filas_a=[0]*(len(data22.columns.values)-1)
    if filas11[i] in filas22:
      aaaa=""
    else:
      aa=filas11[i]
      filas_a.insert(0,aa)
      data22.loc[len(data22)]= filas_a
  for j in range(0,len(filas22)):
    filas_b=[0]*(len(data11.columns.values)-1)
    if filas22[j] in filas11:
      aaaa=""
    else:
      bb=filas22[j]
      filas_b.insert(0,bb)
      data11.loc[len(data11)]= filas_b
  data11.insert(1, "indice", 0)
  data22.insert(1, "indice", 0)
  data111=data11.drop(["Part_%"],axis=1)
  data222=data22.drop(["Part_%"],axis=1)
  for h in range(0,len(data111.iloc[:,1].values)):
    data111.iloc[h,1]=h
  for n in range(0,len(data111.iloc[:,1].values)):
    for m in range(0,len(data222.iloc[:,1].values)):
      if data111.iloc[n,0]==data222.iloc[m,0]:
        data222.iloc[m,1]=data111.iloc[n,1]
  data111=data111.sort_values('indice')
  data222=data222.sort_values('indice')
  data111=data111.set_index(y)
  data222=data222.set_index(y)
  data333=((data222-data111)/(data111))*100
  data333=data333.drop(["indice"],axis=1)
  data333=data333.fillna(0)
  for i in data333.columns:
   data333.loc[data333[i]==np.inf,i] = 100
  sns.heatmap(data333.astype("float"),center=0,cmap='YlGnBu',annot=True, fmt='.0f', annot_kws = {'size': 11},linewidths = .5)


# @title reestablecer variables 2013
def restablecer_variedades_2013(dataframe):
  data_xxx=dataframe.copy()
  for i in range(0,len(data_xxx.columns.values)):
    for j in range(0,len(data_xxx.index.values)):
      if data_xxx.columns[i]=="Variedad":
        ccc=i
  for m in range(0,len(data_xxx.iloc[:,ccc].values)):
    ff=list(data_xxx.iloc[m,ccc])
    for l in range(0,len(ff)):
      if ff[-2]==" ":
        ff.pop((len(ff)-1))
      elif ff[-1]==" ":
        ff.pop((len(ff))-1)
    fff="".join(ff)
    data_xxx.iloc[m,ccc]=fff
  data_nueva=data_xxx.copy()
  list_variedades_nuevas=["ACONCAGUA","ALBA","ALFONSO LAVALLEE","ARIZUL","ASPIRANT BOUSCHET","AURORA","OTRAS VAR. BLANCAS DE MESA","CABERINTA","EMPERATRIZ","GARNACHA (GRENACHE NOIR)","MONASTRELL", "MOSCATUEL","OTRAS VAR. BLANCAS DE PASAS","OTRAS VAR. BLANCAS P/VINIFICAR","OTRAS VAR. NEGRAS DE MESA","OTRAS VAR. NO IDENTIFICADAS","OTRAS VAR. ROSADAS P/PASAS","OTRAS VAR. TINTAS P/VINIFICAR","PASIGA","SYRAH","PINOT GRIS","REINA DE LA VIÑA","RIESLINA","SAUVIGNON BLANC","SUPERIOR SEEDLESS","TINOGASTEÑA","VERDEJO","PERLON","OTRAS VAR. TINTAS P/VINIFICAR"]
  list_variedades_viejas=[415,14984,7085,685,17,11083,2243,12311,43912,831,815,16,50,1497,1579,29,167,4022,50216,7,677,51604,22244,287,646,4803,5608,38205,1525]
  for k in range(0,len(list_variedades_viejas)):
    kkk=data_xxx.iloc[list_variedades_viejas[k],ccc]
    hhh=list_variedades_nuevas[k]
    for j in range(0,len(data_xxx.index)):
      if data_xxx.iloc[j,ccc]==kkk:
        data_nueva.iloc[j,ccc]=hhh
  return data_nueva

# @title reestablecer variables 2018
def restablecer_variedades_2018(dataframe):
  data_xxx=dataframe.copy()
  for i in range(0,len(data_xxx.columns.values)):
    for j in range(0,len(data_xxx.index.values)):
      if data_xxx.columns[i]=="Variedad":
        ccc=i
  for m in range(0,len(data_xxx.iloc[:,ccc].values)):
    ff=list(data_xxx.iloc[m,ccc])
    for l in range(0,len(ff)):
      if ff[-2]==" ":
        ff.pop((len(ff)-1))
      elif ff[-1]==" ":
        ff.pop((len(ff))-1)
    fff="".join(ff)
    data_xxx.iloc[m,ccc]=fff
  data_nueva=data_xxx.copy()
  list_variedades_nuevas=["GARNACHA (GRENACHE NOIR)","OTRAS VAR. BLANCAS DE PASAS","SULTANINA BLANC","VERDEJO","SUPERIOR SEEDLESS"]
  list_variedades_viejas=[7189,5239,33,6617]
  for k in range(0,len(list_variedades_viejas)):
    kkk=data_xxx.iloc[list_variedades_viejas[k],ccc]
    hhh=list_variedades_nuevas[k]
    for j in range(0,len(data_xxx.index)):
      if data_xxx.iloc[j,ccc]==kkk:
        data_nueva.iloc[j,ccc]=hhh
  return data_nueva

# @title reestablecer variables 2022
def restablecer_variedades_2022(dataframe):
  data_xxx=dataframe.copy()
  for i in range(0,len(data_xxx.columns.values)):
    for j in range(0,len(data_xxx.index.values)):
      if data_xxx.columns[i]=="Variedad":
        ccc=i
  for m in range(0,len(data_xxx.iloc[:,ccc].values)):
    ff=list(data_xxx.iloc[m,ccc])
    for l in range(0,len(ff)):
      if ff[-2]==" ":
        ff.pop((len(ff)-1))
      elif ff[-1]==" ":
        ff.pop((len(ff))-1)
    fff="".join(ff)
    data_xxx.iloc[m,ccc]=fff
  data_nueva=data_xxx.copy()
  list_variedades_nuevas=["BLACK SEEDLESS","CRIMSON SEEDLESS","DAWN SEEDLESS","FLAME SEEDLESS","GOLD SEEDLESS","NERO D,AVOLA","RED SEEDLESS","RUBY SEEDLESS","SUPERIOR SEEDLESS"]
  list_variedades_viejas=[4510,35730,43921,44678,46825,66446,74608,74748,77129]
  for k in range(0,len(list_variedades_viejas)):
    kkk=data_xxx.iloc[list_variedades_viejas[k],ccc]
    hhh=list_variedades_nuevas[k]
    for j in range(0,len(data_xxx.index)):
      if data_xxx.iloc[j,ccc]==kkk:
        data_nueva.iloc[j,ccc]=hhh
  return data_nueva

# @title reestablecer variables 2022
def restablecer_provincia_2022(dataframe):
  data_xxx=dataframe.copy()
  for i in range(0,len(data_xxx.columns.values)):
    for j in range(0,len(data_xxx.index.values)):
      if data_xxx.columns[i]=="Provincia":
        ccc=i
  for m in range(0,len(data_xxx.iloc[:,ccc].values)):
    ff=list(data_xxx.iloc[m,ccc])
    for l in range(0,len(ff)):
      if ff[-2]==" ":
        ff.pop((len(ff)-1))
      elif ff[-1]==" ":
        ff.pop((len(ff))-1)
    fff="".join(ff)
    data_xxx.iloc[m,ccc]=fff
  data_nueva=data_xxx.copy()
  list_variedades_nuevas=["BLACK SEEDLESS","CRIMSON SEEDLESS","DAWN SEEDLESS","FLAME SEEDLESS","GOLD SEEDLESS","NERO D,AVOLA","RED SEEDLESS","RUBY SEEDLESS","SUPERIOR SEEDLESS"]
  list_variedades_viejas=[4510,35730,43921,44678,46825,66446,74608,74748,77129]
  for k in range(0,len(list_variedades_viejas)):
    kkk=data_xxx.iloc[list_variedades_viejas[k],ccc]
    hhh=list_variedades_nuevas[k]
    for j in range(0,len(data_xxx.index)):
      if data_xxx.iloc[j,ccc]==kkk:
        data_nueva.iloc[j,ccc]=hhh
  return data_nueva

# @title matriz un filtro
def matriz_un_filtro(dataframe,y):
  data_xxx=dataframe.copy()
  data_xx = pd.pivot_table(data_xxx, index=[y], aggfunc= { 'Superficie_Plantada': 'sum'})
  total=0
  total2=0
  data_nueva=np.zeros((len(data_xx.index)+1,3))
  columnas=[y,"TOTAL","Part_%"]
  data_nueva=pd.DataFrame(data_nueva,columns=columnas).astype("object")  
  
  for i in range(0,len(data_xx.index)):
    data_nueva.iloc[i,0]=data_xx.index[i]
    for j in range(0,len(data_xx.columns)):
      data_nueva.iloc[i,j+1]=data_xx.iloc[i,j]
  data_nueva=data_nueva.fillna(0)
  total=0
  for j in range(0,len(data_xx.index)):
    total= total + data_nueva.iloc[j,1]
  data_nueva.iloc[len(data_xx.index),0]="Total"
  data_nueva.iloc[len(data_xx.index),1]=total

  data_nueva1=data_nueva.copy()
  data_nueva1 = data_nueva1.sort_values("TOTAL", ascending=False)
  data_nueva1=data_nueva1.reset_index()
  data_nueva1=data_nueva1.drop(['index'], axis=1)

  for p in range(0,(len(data_nueva1.iloc[:,0].values))):
    data_nueva1.iloc[p,2] = (data_nueva1.iloc[p,1]/data_nueva1.iloc[0,1])*100

  filas11=[]
  for i in data_nueva1.iloc[:,0].values:
    gg=list(i)
    for j in range(0,len(gg)):
      if gg[-2]==" ":
        gg.pop((len(gg)-1))
      elif gg[-1]==" ":
        gg.pop((len(gg))-1)
    ggg="".join(gg)
    filas11.append(ggg)
  data_nueva1.iloc[:,0]=filas11

  return data_nueva1

# @title matiz doble filtro
def matriz_doble_filtro_2(dataframe,x,y):
  data_xxx=dataframe.copy()
  data_xx = pd.pivot_table(data_xxx, index=[y], columns=[x], aggfunc= { 'Superficie_Plantada': 'sum'})
  total=0
  total2=0
  data_nueva=np.zeros((len(data_xx.index)+1,len(data_xx.columns)+1+2))
  columnas=[y,"TOTAL","Part_%"]
  columnas1=[y,"TOTAL","Part_%"]
  listafiltro2=[]
  base_columnas=list(data_xx.columns)
  for z in base_columnas:
    filtro_columna=list(z)
    del filtro_columna[0]
    listafiltro2.append(filtro_columna[0])
  for f in range(0,len(data_xx.columns)):
    ll=listafiltro2[f]
    columnas.insert(f+1,ll)
  data_nueva=pd.DataFrame(data_nueva,columns=columnas)
  data_nueva.iloc[0] = data_nueva.iloc[0].astype(str)

  for i in range(0,len(data_xx.index)):
    data_nueva.iloc[i,0]=data_xx.index[i]
    for j in range(0,len(data_xx.columns)):
      data_nueva.iloc[i,j+1]=data_xx.iloc[i,j]
  data_nueva=data_nueva.fillna(0)

  for i in range(0,len(listafiltro2)):
    total=0
    for j in range(0,len(data_xx.index)):
      total= total + data_nueva.iloc[j,i+1]
    data_nueva.iloc[len(data_xx.index),0]="Total"
    data_nueva.iloc[len(data_xx.index),i+1]=total

  for p in range(0,len(data_xx.index)+1):
    data_nueva.iloc[p,(len(listafiltro2)+1)]=0
    for q in range(0,(len(listafiltro2))):
      data_nueva.iloc[p,(len(listafiltro2)+1)] = data_nueva.iloc[p,(len(listafiltro2)+1)]+  data_nueva.iloc[p,(len(listafiltro2)+1-q-1)]

  data_nueva1=data_nueva.copy()
  data_nueva1 = data_nueva1.sort_values("TOTAL", ascending=False)
  data_nueva1=data_nueva1.reset_index()
  data_nueva1=data_nueva1.drop(['index'], axis=1)

  for p in range(0,(len(data_xx.index)+1)):
    data_nueva1.iloc[p,len(listafiltro2)+2] = data_nueva1.iloc[p,len(listafiltro2)+1]/data_nueva1.iloc[0,len(listafiltro2)+1]
  columnas11=[]
  for i in data_nueva1.columns.values:
    ff=list(i)
    for j in range(0,len(ff)):
      if ff[-2]==" ":
        ff.pop((len(ff)-1))
      elif ff[-1]==" ":
        ff.pop((len(ff))-1)
    fff="".join(ff)
    columnas11.append(fff)
  data_nueva1.columns=columnas11

  filas11=[]
  for i in data_nueva1.iloc[:,0].values:
    gg=list(i)
    for j in range(0,len(gg)):
      if gg[-2]==" ":
        gg.pop((len(gg)-1))
      elif gg[-1]==" ":
        gg.pop((len(gg))-1)
    ggg="".join(gg)
    filas11.append(ggg)
  data_nueva1.iloc[:,0]=filas11
  return data_nueva1

# @title matriz_corta
def matriz_corta(dataxx,rr):
  data_nueva=dataxx.iloc[0:(len(dataxx.index)-rr)].copy()
  for i in range(0,len(dataxx.iloc[0,:])):
    sumar=0
    for j in range(0,len(dataxx.iloc[:,0])):
      if j<((len(dataxx.index)-rr)-3):
        data_nueva.iloc[j,i]=dataxx.iloc[j,i]
      if j>((len(dataxx.index)-rr)-2):
        if i!=0:
          sumar=sumar+dataxx.iloc[j,i]
    if i==0:
      data_nueva.iloc[((len(dataxx.index)-rr)-1),i]="Otras Provincias"
    else:
     data_nueva.iloc[((len(dataxx.index)-rr)-1),i]=sumar

  return data_nueva

# @title matriz evolucion un filtro
def matriz_un_filtro_evolucion(dataserie,años,y):
  data_nueva=matriz_un_filtro(dataserie[len(dataserie)-1],y)
  data_nueva.rename(columns={'TOTAL':años[len(años)-1]},inplace=True)
  for l in range(0,len(dataserie)-1):
    dataxx=matriz_un_filtro(dataserie[l],y)
    data_nueva.insert(l+1,años[l],0)
    for i in range(0,len(data_nueva.index)):
      for j in range(0,len(dataxx.index)):
        if data_nueva.iloc[i,0]==dataxx.iloc[j,0]:
          data_nueva.iloc[i,l+1]=dataxx.iloc[j,1].astype('float64')      
  return data_nueva

# @title doble grafico
def doble_grafico_part_evol(dataframe):
  data_xxxx=dataframe.copy()


  plt.figure(figsize=(12,3))

  plt.subplot(1,2,1)
  explode=[0]*(len(data_xxxx.index)-1)
  for i in range(0,len(explode)):
    if data_xxxx.iloc[i,(len(data_xxxx.columns)-1)]>6:
      explode[i]=0
    else:
      explode[i]=0.5
  plt.pie(data_xxxx.iloc[1:,(len(data_xxxx.columns)-1)].values,autopct="%0.0f %%",pctdistance=1.15,explode= explode ,textprops={'size': 'smaller'})
  plt.title("Participación de la Superficie", size=15)
  plt.legend(labels=data_xxxx.iloc[1:,0].values,loc="upper center",
           bbox_to_anchor=(0.5, -0.001), ncol=2)

  plt.subplot(1,2,2)
  for i in range(0,len(data_xxxx.index)):
    plt.plot(data_xxxx.columns[1:(len(data_xxxx.columns))-1].values, data_xxxx.iloc[i,1:(len(data_xxxx.columns))-1],label=data_xxxx.iloc[i,0])
  plt.title("Evolución de la Superficie Vitivinícola", size=15)
  plt.legend(labels=data_xxxx.iloc[:,0].values,loc="upper center",
           bbox_to_anchor=(0.5, -0.1), ncol=3)

# @title matriz un filtro evolucion con variable
def matriz_evolucion_con_variacion(dataframe,años):
  dataz=dataframe.copy()
  aa="var_"+años[-1]+"/"+años[0]
  bb="var_"+años[-1]+"/"+años[-2]
  dataz.insert(len(dataz.columns),aa,0)
  dataz.insert(len(dataz.columns),bb,0)
  for i in range(0,len(dataz.index.values)):
    dataz.iloc[i,len(dataz.columns)-2]=(((dataz.iloc[i,len(dataz.columns)-4]-dataz.iloc[i,1]))/(dataz.iloc[i,1]))*100
    dataz.iloc[i,len(dataz.columns)-1]=(((dataz.iloc[i,len(dataz.columns)-4]-dataz.iloc[i,len(dataz.columns)-5]))/(dataz.iloc[i,len(dataz.columns)-5]))*100
  return dataz

# @title grafico evolucion porcental en heatmap
def graf_evol_porcent_y_heat(dataserie,años,y):
  data_nueva=matriz_un_filtro_evolucion(dataserie,años,y)
  data_nueva1=data_nueva.copy()
  data_xxxx=data_nueva1.iloc[:,0:(len(data_nueva1)-1)].copy()
  data_base=data_xxxx.copy()
  for i in range(0,len(data_xxxx.index.values)):
    for j in range(0,len(data_xxxx.columns.values)-1):
      data_xxxx.iloc[i,j+1]= ((data_base.iloc[i,j+1])/(data_base.iloc[i,1]))*100

  data_nueva2=data_xxxx.set_index(y)
  sns.heatmap(data_nueva2,center=0, cmap='YlGnBu',annot=True, fmt='.1f', annot_kws = {'size': 12},linewidths=1)
  plt.tick_params(labelsize=10, length=0, width=-1000.1, grid_color='r', grid_alpha=0.2, direction='out', pad=0.0001, left=True)
  plt.gcf().set_size_inches(6, 3)
  plt.xticks(rotation=45)
  plt.show()

# @title grafico evolucion porcentual
def graf_evol_porcent(dataseries,años,y):
  data_nueva=matriz_un_filtro_evolucion(dataseries,años,y)
  data_xxxx=data_nueva.iloc[:,0:5].copy()
  for i in range(0,len(data_xxxx.index)):
    if data_xxxx.iloc[i,1]==0:
      data_xxxx.iloc[i,1]=1
  data_nueva2=data_xxxx.copy()
  for i in range(0,len(data_xxxx.index)):
    for j in range(0,len(data_xxxx.columns)-1):
      data_nueva2.iloc[i,j+1]=0
  for i in range(0,len(data_xxxx.index)):
    for j in range(0,len(data_xxxx.columns)-1):
      data_nueva2.iloc[i,j+1]=data_xxxx.iloc[i,j+1]/data_xxxx.iloc[i,1]

  data_nueva2=data_xxxx.set_index(y)

  sns.heatmap(data_nueva2,center=0, annot=True, fmt='.1f', annot_kws = {'size': 12},linewidths=1)
  plt.tick_params(labelsize=10, length=0, width=-1000.1, grid_color='r', grid_alpha=0.2, direction='out', pad=0.0001, left=True)
  plt.gcf().set_size_inches(6, 3)
  plt.xticks(rotation=45)
  plt.show()

# @title grafico evol heatmap
def graf_evol_porcent_heatmap(dataseries,años,y,rr):
  data_xx=matriz_un_filtro_evolucion(dataseries,años,y)
  data_xxxx=data_xx.iloc[:rr,0:5].copy()
  data_new=data_xx.iloc[:rr,0:5].copy()
  for i in range(0,len(data_xxxx.index)):
    if data_xxxx.iloc[i,1]==0:
      data_xxxx.iloc[i,1]=1
  for i in range(0,len(data_new.index)):
    for j in range(0,len(data_new.columns)-1):
      data_new.iloc[i,j+1]=0
  for i in range(0,len(data_xxxx.index)):
    for j in range(0,len(data_xxxx.columns)-1):
      data_new.iloc[i,j+1]=data_xxxx.iloc[i,j+1]/data_xxxx.iloc[i,1]*100
  data_nueva=data_new.set_index(y)

  sns.heatmap(data_nueva,center=0,annot=True, fmt='.1f', annot_kws = {'size': 12},linewidths=1)
  plt.tick_params(labelsize=10, length=0, width=-1000.1, grid_color='r', grid_alpha=0.2, direction='out', pad=0.0001, left=True)
  plt.gcf().set_size_inches(8, 5)
  plt.xticks(rotation=45)
  plt.show()

# @title grafico_evoluc_linea
def graf_evol_porcent_linea(dataseries,años,y):
  data_xx=matriz_un_filtro_evolucion(dataseries,años,y)
  data_xxxx=data_xx.iloc[:,0:(len(años)+1)].copy()
  data_new=data_xx.iloc[:,0:(len(años)+1)].copy()
  for i in range(0,len(data_xxxx.index)):
    if data_xxxx.iloc[i,1]==0:
      data_xxxx.iloc[i,1]=1
  for i in range(0,len(data_new.index)):
    for j in range(0,len(data_new.columns)-1):
      data_new.iloc[i,j+1]=0
  for i in range(0,len(data_xxxx.index)):
    for j in range(0,len(data_xxxx.columns)-1):
      data_new.iloc[i,j+1]=data_xxxx.iloc[i,j+1]/data_xxxx.iloc[i,1]*100
  data_nueva=data_new.set_index(y)

  for i in range(0,len(data_nueva.index)):
    plt.plot(data_nueva.columns.values, data_nueva.iloc[i,0:len(años)],label=data_nueva.index.values)
  plt.title("Evolución de la Superficie Vitivinícola", size=15)
  plt.legend(labels=data_nueva.index.values,loc="upper center",
           bbox_to_anchor=(0.5, -0.1), ncol=3)

# @title grafico
def doble_grafico_part_heat_doble_filtro(dataframe,y):
  data_xxxx=dataframe.copy()

  plt.figure(figsize=(12,3))

  plt.subplot(1,2,1)
  explode=[0]*(len(data_xxxx.index)-1)
  for i in range(0,len(explode)):
    if data_xxxx.iloc[i,(len(data_xxxx.columns)-1)]>15:
      explode[i]=0
    else:
      explode[i]=0.5
  plt.pie(data_xxxx.iloc[1:,(len(data_xxxx.columns)-1)].values,autopct="%0.0f %%",pctdistance=1.15,explode= explode ,textprops={'size': 'smaller'})
  plt.title("Participación de la Superficie", size=15)
  plt.legend(labels=data_xxxx.iloc[1:,0].values,loc="upper center",
           bbox_to_anchor=(0.5, -0.001), ncol=2)

  plt.subplot(1,2,2)
  data_xx=dataframe.copy()
  data_base=data_xx.copy()
  for i in range(0,len(data_xx.index.values)):
    sumar=0
    for j in range(0,len(data_xx.columns.values)-2):
      sumar= sumar + data_xx.iloc[i,j+1]
    data_xx.iloc[i,(len(data_xx.columns)-1)] = ((data_base.iloc[i,j+1])/sumar)*100

  data_nueva2=data_xx.set_index(y)
  sns.heatmap(data_nueva2.astype(float),center=0, annot=True, fmt='.1f', annot_kws = {'size': 12},linewidths=1)
  plt.tick_params(labelsize=10, length=0, width=-1000.1, grid_color='r', grid_alpha=0.2, direction='out', pad=0.0001, left=True)
  plt.gcf().set_size_inches(10, 3)
  plt.xticks(rotation=45)
  plt.show()

def matriz_doble_filtro_evolucion(dataserie,años,y,x,lugar):
  data_nueva=matriz_doble_filtro_2(dataserie[len(dataserie)-1],y,x)
  base_columnas=[]
  for i in range(0,len(data_nueva.columns)):
    if (data_nueva.columns[i]!=lugar):
      if data_nueva.columns[i]!=x :
        base_columnas.append(data_nueva.columns[i])
  data_nueva=data_nueva.drop(base_columnas,axis=1)
  data_nueva.rename(columns={lugar:años[len(años)-1]},inplace=True)
  for l in range(0,len(dataserie)-1):
    data_nueva.insert(l+1,años[l],0)
    base_columnas=[]
    dataxx= matriz_doble_filtro_2(dataserie[l],y,x)
    for i in range(0,len(dataxx.columns)):
      if (dataxx.columns[i]!=lugar):
        if dataxx.columns[i]!=x :
          base_columnas.append(dataxx.columns[i])
    dataxxx=dataxx.drop(base_columnas,axis=1)
    for i in range(0,len(data_nueva.index.values)):
      for j in range(0,len(dataxxx.index.values)):
        if data_nueva.iloc[i,0]==dataxxx.iloc[j,0]:
          data_nueva.iloc[i,l+1]=dataxxx.iloc[j,1]

  data_nueva.insert(len(data_nueva.columns),"Part_%",0)
  for i in range(0,len(data_nueva.index)):
    data_nueva.iloc[i,len(data_nueva.columns)-1] =data_nueva.iloc[i,len(data_nueva.columns)-2]/data_nueva.iloc[0,len(data_nueva.columns)-2]*100
  return data_nueva

# @title doble grafico evolucion y barras
def doble_grafico_part_evol_barras(dataframe):
  data_xxxx=dataframe.copy()

  plt.figure(figsize=(12,3))

  plt.subplot(1,2,1)
  explode=[0]*(len(data_xxxx.index)-1)
  for i in range(0,len(explode)):
    if data_xxxx.iloc[i,(len(data_xxxx.columns)-1)]>4:
      explode[i]=0
    else:
      explode[i]=0.2
  plt.pie(data_xxxx.iloc[1:,(len(data_xxxx.columns)-1)].values,autopct="%0.0f %%",pctdistance=1.15,explode= explode ,textprops={'size': 'smaller'})
  plt.title("Participación de la Superficie", size=15)
  plt.legend(labels=data_xxxx.iloc[1:,0].values,loc="upper center",
           bbox_to_anchor=(0.5, -0.001), ncol=2)

  plt.subplot(1,2,2)
  plt.bar(data_xxxx.columns[1:5], data_xxxx.iloc[1,1:5] + data_xxxx.iloc[2,1:5] + data_xxxx.iloc[3,1:5],label=data_xxxx.iloc[3,0])
  plt.bar(data_xxxx.columns[1:5], data_xxxx.iloc[1,1:5] + data_xxxx.iloc[2,1:5],label=data_xxxx.iloc[2,0])
  plt.bar(data_xxxx.columns[1:5], data_xxxx.iloc[1,1:5],label=data_xxxx.iloc[1,0])
  plt.title("Evolución de la Superficie Vitivinícola", size=15)
  plt.legend(loc="upper center",
           bbox_to_anchor=(0.5, -0.1), ncol=2)

# @title matriz variacion con doble filtro
def matriz_doble_filto_variacion_2_años(dataframe,dataframe2,xx,yy):
  datazz=matriz_doble_filtro_2(dataframe,xx,yy)
  datayy=matriz_doble_filtro_2(dataframe2,xx,yy)
  for i in range(0,len(datazz.columns.values)):
    if datazz.columns[i] not in datayy.columns.values:
      datayy.insert((len(datayy.columns)),datazz.columns[i],0)
  for i in range(0,len(datayy.columns.values)):
    if datayy.columns[i] not in datazz.columns.values:
      datazz.insert((len(datazz.columns)),datayy.columns[i],0)

  for i in range(0,len(datazz.index)):
    contar=0
    filas_a=[0]*(len(datayy.columns.values)-1)
    if datazz.iloc[i,0] in datayy.iloc[:,0].values:
      aa=""
    else:
      aaa=datazz.iloc[i,0]
      filas_a.insert(0,aaa)
      datayy.loc[len(datayy)]= filas_a

  for i in range(0,len(datayy.index)):
    filas_b=[0]*(len(datazz.columns.values)-1)
    if datayy.iloc[i,0] in datazz.iloc[:,0].values:
      bb=""
    else:
      bbb=datayy.iloc[i,0]
      filas_b.insert(0,bbb)
      datazz.loc[len(datazz)]= filas_b

  datazz.insert(1, "indice", 0)
  datayy.insert(1, "indice", 0)
  datazzz=datazz.drop(["Part_%"],axis=1)
  datayyy=datayy.drop(["Part_%"],axis=1)

  for h in range(0,len(datazzz.iloc[:,1].values)):
    datazzz.iloc[h,1]=h
  for n in range(0,len(datazzz.iloc[:,1].values)):
    for m in range(0,len(datayyy.iloc[:,1].values)):
      if datazzz.iloc[n,0]==datayyy.iloc[m,0]:
        datayyy.iloc[m,1]=datazzz.iloc[n,1]
  datazzz=datazzz.sort_values('indice')
  datayyy=datayyy.sort_values('indice')
  datazzz=datazzz.set_index(yy)
  datayyy=datayyy.set_index(yy)

  data333=(((datayyy-datazzz)/(datazzz))*100)
  data333=data333.drop(["indice"],axis=1)
  data333=data333.fillna(0.01)
  for i in data333.columns:
   data333.loc[data333[i]==np.inf,i] = 100
  return data333

# @title matriz corta sin suma final
def matriz_corta_sin_suma(dataframe,rr1,rr2):
  dataxx1=dataframe.copy()
  dataxx=dataxx1.reset_index()
  data_nueva1=dataframe.iloc[rr1:rr2,:].copy()
  return data_nueva1

# @title Texto de título predeterminado
def doble_grafico_part_evol_barras(dataframe):
  data_xxxx=dataframe.copy()

  plt.figure(figsize=(12,3))

  plt.subplot(1,2,1)
  explode=[0]*(len(data_xxxx.index)-1)
  for i in range(0,len(explode)):
    if data_xxxx.iloc[i,(len(data_xxxx.columns)-1)]>4:
      explode[i]=0
    else:
      explode[i]=0.2
  plt.pie(data_xxxx.iloc[1:,(len(data_xxxx.columns)-1)].values,autopct="%0.0f %%",pctdistance=1.15,explode= explode ,textprops={'size': 'smaller'})
  plt.title("Participación de la Superficie", size=15)
  plt.legend(labels=data_xxxx.iloc[1:,0].values,loc="upper center",
           bbox_to_anchor=(0.5, -0.001), ncol=2)

  plt.subplot(1,2,2)


  plt.title("Evolución de la Superficie Vitivinícola", size=15)
  for i in range(0,len(data_xxxx.index)-1):
    plt.bar(data_xxxx.columns[1:(len(data_xxxx.columns)-1)], data_xxxx.iloc[i+1,1:(len(data_xxxx.columns)-1)], bottom = np.sum(data_xxxx.iloc[1:i+1,1:(len(data_xxxx.columns)-1)], axis = 0),label=data_xxxx.iloc[i+1,0])
  plt.legend(loc="upper center",
           bbox_to_anchor=(0.5, -0.1), ncol=3)

# @title matriz variacion con doble filtro
def matriz_doble_filto_variacion_2_años(dataframe,dataframe2,filtro2,filtro3):
  datazz=matriz_doble_filtro_2(dataframe,filtro2,filtro3)
  datayy=matriz_doble_filtro_2(dataframe2,filtro2,filtro3)
  for i in range(0,len(datazz.columns.values)):
    if datazz.columns[i] not in datayy.columns.values:
      datayy.insert((len(datayy.columns)),datazz.columns[i],0)
  for i in range(0,len(datayy.columns.values)):
    if datayy.columns[i] not in datazz.columns.values:
      datazz.insert((len(datazz.columns)),datayy.columns[i],0)

  for i in range(0,len(datazz.index)):
    contar=0
    filas_a=[0]*(len(datayy.columns.values)-1)
    if datazz.iloc[i,0] in datayy.iloc[:,0].values:
      aa=""
    else:
      aaa=datazz.iloc[i,0]
      filas_a.insert(0,aaa)
      datayy.loc[len(datayy)]= filas_a

  for i in range(0,len(datayy.index)):
    filas_b=[0]*(len(datazz.columns.values)-1)
    if datayy.iloc[i,0] in datazz.iloc[:,0].values:
      bb=""
    else:
      bbb=datayy.iloc[i,0]
      filas_b.insert(0,bbb)
      datazz.loc[len(datazz)]= filas_b

  datazz.insert(1, "indice", 0)
  datayy.insert(1, "indice", 0)
  datazzz=datazz.drop(["Part_%"],axis=1)
  datayyy=datayy.drop(["Part_%"],axis=1)

  for h in range(0,len(datazzz.iloc[:,1].values)):
    datazzz.iloc[h,1]=h
  for n in range(0,len(datazzz.iloc[:,1].values)):
    for m in range(0,len(datayyy.iloc[:,1].values)):
      if datazzz.iloc[n,0]==datayyy.iloc[m,0]:
        datayyy.iloc[m,1]=datazzz.iloc[n,1]
  datazzz=datazzz.sort_values('indice')
  datayyy=datayyy.sort_values('indice')
  datazzz=datazzz.set_index(filtro3)
  datayyy=datayyy.set_index(filtro3)

  data333=(((datayyy-datazzz)/(datazzz))*100)
  data333=data333.drop(["indice"],axis=1)
  data333=data333.fillna(0.01)
  for i in data333.columns:
    data333.loc[data333[i]==np.inf,i] = 100
  return data333

def matriz_doble_filto_evolucion(dataserie,años,y,x,lugar):
  data_nueva=matriz_doble_filtro_2(dataserie[len(dataserie)-1],y,x)
  base_columnas=[]
  for i in range(0,len(data_nueva.columns)):
    if (data_nueva.columns[i]!=lugar):
      if data_nueva.columns[i]!=x :
        base_columnas.append(data_nueva.columns[i])
  data_nueva=data_nueva.drop(base_columnas,axis=1)
  data_nueva.rename(columns={lugar:años[len(años)-1]},inplace=True)
  for l in range(0,len(dataserie)-1):
    data_nueva.insert(l+1,años[l],0)
    base_columnas=[]
    dataxx= matriz_doble_filtro_2(dataserie[l],y,x)
    for i in range(0,len(dataxx.columns)):
      if (dataxx.columns[i]!=lugar):
        if dataxx.columns[i]!=x :
          base_columnas.append(dataxx.columns[i])
    dataxxx=dataxx.drop(base_columnas,axis=1)
    for i in range(0,len(data_nueva.index.values)):
      for j in range(0,len(dataxxx.index.values)):
        if data_nueva.iloc[i,0]==dataxxx.iloc[j,0]:
          data_nueva.iloc[i,l+1]=dataxxx.iloc[j,1]

  data_nueva.insert(len(data_nueva.columns),"Part_%",0)
  for i in range(0,len(data_nueva.index)):
    data_nueva.iloc[i,len(data_nueva.columns)-1] =data_nueva.iloc[i,len(data_nueva.columns)-2]/data_nueva.iloc[0,len(data_nueva.columns)-2]*100
  return data_nueva

def doble_grafico_part_evol_barras(dataframe):
  data_xxxx=dataframe.copy()

  plt.figure(figsize=(12,3))

  plt.subplot(1,2,1)
  explode=[0]*(len(data_xxxx.index)-1)
  for i in range(0,len(explode)):
    if data_xxxx.iloc[i,(len(data_xxxx.columns)-1)]>4:
      explode[i]=0
    else:
      explode[i]=0.2
  plt.pie(data_xxxx.iloc[1:,(len(data_xxxx.columns)-1)].values,autopct="%0.0f %%",pctdistance=1.15,explode= explode ,textprops={'size': 'smaller'})
  plt.title("Participación de la Superficie", size=15)
  plt.legend(labels=data_xxxx.iloc[1:,0].values,loc="upper center",
           bbox_to_anchor=(0.5, -0.001), ncol=2)

  plt.subplot(1,2,2)
  columnas=[]
  for i in range(0,len(data_xxxx.columns)-2):
    columnas.append(data_xxxx.columns[i+1])
  for i in range(0,len(data_xxxx.index)-1):
    plt.bar(columnas, data_xxxx.iloc[i+1,1:len(data_xxxx.columns)-1], bottom = np.sum(data_xxxx.iloc[1:i+1,1:len(data_xxxx.columns)-1] ,axis=0),label=data_xxxx.iloc[i+1,0])
  plt.title("Evolución de la Superficie Vitivinícola", size=15)
  plt.legend(loc="upper center",
           bbox_to_anchor=(0.5, -0.1), ncol=2)
  
  # @title graficar heatmap var viejo
def graf_heatmap_var(dataframe,dataframe2,x,y):
  data11=matriz_doble_filtro_2(dataframe,x,y)
  data22=matriz_doble_filtro_2(dataframe2,x,y)
  datazz=matriz_doble_filto_variacion_2_años(dataframe,dataframe2)
  fig, ax = plt.subplots()
  sns.heatmap(datazz,center=0,cmap='YlGnBu',annot=True, fmt='.0f', annot_kws = {'size': 11},linewidths = .5)
  # Show all ticks and label them with the respective list entries
  ax.set_yticks(np.arange(len(data11[y])), labels=data11[y].values)
  plt.yticks(rotation=0)
  plt.tick_params(labelsize="small", length=0, width=-1000.1, grid_color='r', grid_alpha=0.2, direction='out', pad=0.0001, left=True)
  plt.gcf().set_size_inches(10, 5)
  plt.show()