import dtale
import dtale.app as dtale_app

from Extract import extract_dfs
from Explore import explore_dfs
from Transform import transform
from Clustering import clustering

df_dic = extract_dfs()

explore_dfs(df_dic)

df_cluster = transform(df_dic)

clustering(df_cluster)