import pandas as pd
import plotly.express as px
import plotly.io as pio
import cufflinks as cf
import plotly.offline as offline

df = pd.read_csv('../data/final2.csv', encoding='euc-kr')

df[['value_1', 'value_2', 'value_3', 'value_4', 'value_5', 'value_6' ]] = df[['value_1', 'value_2', 'value_3', 'value_4', 'value_5', 'value_6' ]].round().astype(int)

mapping = {0: '양호', 1: '경증', 2: '중등도', 3: '중증'}

# 'value_1'부터 'value_6'까지의 열에 매핑 적용
df[['value_1', 'value_2', 'value_3', 'value_4', 'value_5', 'value_6']] = df[['value_1', 'value_2', 'value_3', 'value_4', 'value_5', 'value_6']].map(mapping.get)





# 각 value_1 값에 대한 카운트
#value_1_counts = df['value_1'].value_counts()

#fig = px.pie(names=value_1_counts.index, values=value_1_counts,
#             color_discrete_sequence=px.colors.sequential.RdBu)
#fig.update_traces(textposition='inside', textinfo='percent+label', textfont=dict(size=14))

# HTML로 렌더링하고 표시
#offline.plot(fig, filename='example_plot.html', auto_open=True)


#fig1.write_html("fig1.html", include_mathjax= 'cdn')

def graphmaker(owngender, ownage, p_values):

    for index, value in enumerate(p_values):
        if value > 0:
            kind = df[(df['gender'] == owngender)&(df['age'] == ownage)]['value_' + str(index+1)].value_counts()

            fig = px.pie(names= kind.index, values= kind,
                         color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_traces(textposition='inside', textinfo='percent+label', textfont=dict(size=14))

            fig.write_html(owngender+ownage+"value_"+str(index+1)+"chart.html", include_mathjax='cdn')

graphmaker('남', '10대', [1,0,0,0,0,0])





