import pandas as pd
import numpy as np
import glob
import random as r

# set_correct_thres = 0.5

class Submission():
    def __init__(self, df, model_name):
        self.df = df
        self.model_name = model_name
        self.mean_score = -1
        self.score_list = []
        self.set_correct_thres = 0.5
        self.pred_rate = -1
    
    def pre_process(self):
        col_good, col_ok, col_non, col_pred = 'eff_sol', 'half_eff_sol', 'non_eff_sol', 'sol_pred'
        df_no = self.df[self.df[col_pred]=='No']
        self.df = self.df[self.df[col_pred]!='No'].reset_index(drop=True)
        self.pred_rate = 1-(len(df_no)/len(self.df))
        
        self.df['sol_good'] = self.df[col_good].map(lambda x: eval(x) if type(x) == str else [])
        self.df['sol_ok'] = self.df[col_ok].map(lambda x: eval(x) if type(x) == str else [])
        self.df['sol_non'] = self.df[col_non].map(lambda x: eval(x) if type(x) == str else [])
        self.df['sol_pred'] = self.df[col_pred].map(lambda x: eval(x) if type(x) == str else [])

    def score_one_row(self, good_point, ok_point, bad_point, row_i):
        dict_weighted_eff = dict()
        list_good = list(self.df['sol_good'][row_i])
        list_ok = list(self.df['sol_ok'][row_i])
        list_non = list(self.df['sol_non'][row_i])

        all_sol_used = [] 
        all_sol_used.extend(list_good)
        all_sol_used.extend(list_ok)
        all_sol_used.extend(list_non)
        all_sol_used = list(set(all_sol_used))
        list_pred = list(self.df['sol_pred'][row_i])
        
        # 算各個solution code的加權效度存於 dict_weighted_eff
        for s in all_sol_used:
            total_n = list_good.count(s)+list_ok.count(s)+list_non.count(s)
            dict_weighted_eff[s] = (list_good.count(s)*good_point + list_ok.count(s)*ok_point)/total_n
        
        # 計算正解
        sol_accurate = [s for s in all_sol_used if dict_weighted_eff[s]>=self.set_correct_thres]
        
        # 計算CA
        full_score = sum([dict_weighted_eff[s] for s in sol_accurate])
        list_in_sol = [s for s in list_pred if s in sol_accurate]
        list_out_sol = [s for s in list_pred if s not in sol_accurate]
        
        dict_score = dict()
        dict_score['list_good'] = list_good
        dict_score['list_ok'] = list_ok
        dict_score['list_non'] = list_non
        dict_score['dict_weighted_eff'] = dict_weighted_eff
        dict_score['sol_accurate'] = sol_accurate
        dict_score['full_score'] = full_score
        dict_score['list_in_sol'] = list_in_sol
        dict_score['list_out_sol'] = list_out_sol
        
        try:
            #預測到哪一個就得到他的分數 - 多預測的分數
            pred_score = (sum([dict_weighted_eff[s] for s in list_in_sol])+bad_point*len(list_out_sol))/full_score
            score_row_i = max(round(pred_score,4),0)
        except:
            score_row_i = np.nan
        return dict_score, score_row_i
    
    def score(self, good_point, ok_point, bad_point):
        self.df['score_res'] = [self.score_one_row(good_point, ok_point, bad_point, row_i) for row_i in range(len(self.df))]
        self.df['score'] = self.df['score_res'].map(lambda x: max(x[1],0))
        
    def print_res(self):
        try:
            self.score_list = self.df['score']
        except:
            'Please tried to score first.'
        self.mean_score = self.score_list.mean()
        print('============== Score of ('+self.model_name+') ==============')
        print('CA Score = ', round(self.mean_score,4))
        return self.mean_score
        
    def CA_Score(self,good_point, ok_point, bad_point):
        self.pre_process()
        self.score(good_point, ok_point, bad_point)
        ca_score = self.print_res()
        return ca_score
    
     #單獨計算某個PID預測分數的功能
    def score_one_pid(self, pid):
        try:
            return self.df[self.df['PID']==pid]['score_res'].values[0]
        except:
            print('Try another PID...')

if _name_ == __main__:
    path = 'submission\\'
    file_name = 'model_A_result'
    df = pd.read_csv(path+file_name+'.csv', encoding='big5')

    good_point, ok_point, bad_point = 1, 0.5, -0.01

    sub = Submission(df,'model_A')
    ca_score = sub.CA_Score(good_point, ok_point, bad_point)
    print('ca_score = ',ca_score)