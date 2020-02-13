import pandas as pd
import spacy
from spacy.lang.en import English
nlp = English()
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

#your SM output
df=pd.read_csv('Downloads/as_culture_writeins - Sheet1 (1).csv')

gen=[]
for i in list(df['What is your age?']):
    if i < 24:
        gen.append('Generation Z')
    elif 24 <= i <= 39:
        gen.append('Millennial')
    elif 40<=i<=56:
        gen.append('Generation X')
    else:
        gen.append('Baby Boomer or Older')
df['Generation']=gen

#make a list of columns where you want to separate out words
targets = ['ask','meaning']

for target_word in targets:
    cols=list(df)
    for w in targets:
        if w != target_word:
            cols.remove(w)
    df_a=df[cols]
    df_a=df_a.dropna(subset=[target_word])
    df_a_list= df_a.to_dict('records')
    asks=list(df_a[target_word])
    for i, row in enumerate(asks):
        doc = nlp(row)
        tokens = [token.text for token in doc if not token.is_stop]
        tokens = [x for x in tokens if len(x) > 1]
        asks[i]=tokens
    out=[]
    terms=[]
    for i, person in enumerate(df_a_list):
        if asks[i]:
            for word in range(len(asks[i])):
                out.append(person)
                terms.append(asks[i][word])
        else:
            terms.append(np.nan)
            out.append(person)
    df_ask=pd.DataFrame(out)
    terms = [str(x).lower() for x in terms]
    df_ask['word']=terms
    df_ask.to_csv(target_word+'_out.csv')
