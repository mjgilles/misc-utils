import time
import tweepy
import requests
auth = tweepy.OAuthHandler("###", "###")
auth.set_access_token("###", 
                      "###")
api = tweepy.API(auth)
import pandas as pd
df=pd.read_csv('list-of-handles.csv')
handles=[x.split(" ")[-1] for x in list(df.twitter)]
users=[]
image_urls=[]
tells=[x*100 for x in range(1000)]
paths=[]
extensions=['jpg','jpeg','png']
import numpy as np
for i,h in enumerate(handles[:10]):
    if h not in skips:
    try:
        v=api.get_user(h)
        image_target=v.profile_image_url.split("_normal")[0]+'_400x400.'
        for ext in extensions:
            img_data = requests.get(image_target+ext)
            if img_data.status_code == 200:
                print (image_target+ext)
                break
            else:
                pass
        with open('cg_t_avatars/'+h+'.jpg', 'wb') as handler:
            handler.write(img_data.content)
        image_urls.append(image_target+ext)
        #print (image_target)
        paths.append('./cg_t_avatars/'+h+'.jpg')
    except:
        image_urls.append(np.nan)
        paths.append(np.nan)
    users.append(h)
    time.sleep(1.1)
    if i in tells:
        print ("Completed "+str(i))
    else:
        pass
tdf=pd.DataFrame({'twitter':users,'image_urls':image_urls,'paths':paths})
out_df=pd.merge(df,tdf.dropna(),on='twitter')
out_df.to_csv('###.csv')
