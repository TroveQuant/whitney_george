from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import time
import math
import tinyshare as tu
import base64
import email_sender_v2
import a_passwards as pw

# --- 全局显示与绘图设置 ---
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.width', 180)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# --- 路径与 API 设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, 'data')
if not os.path.exists(data_path):
    os.makedirs(data_path)
os.chdir(data_path)

# 设置 Token
tu.set_token('4dfe55ae66614ca943e09a6d82339eb65b77dcaf327841ba3d5c1574')
pro = tu.pro_api()

# --- 全局时间参数 ---
data_start = '20171231'  # 数据起始日期（需比实际回测开始时间提前2个季度以获取财报）
calc_start_date = pd.to_datetime(data_start) + pd.DateOffset(months=6)
start_date = calc_start_date.strftime('%Y%m%d')

def get_financial_dates(today_str):
    """
    计算财报数据获取的时间窗口
    逻辑: data_end1 为当前季度往前推 2 个季度的季度末
    """
    current_dt = pd.to_datetime(today_str)
    current_quarter = pd.Period(current_dt, freq='Q')
    target_quarter = current_quarter - 2
    d1_date = target_quarter.end_time.date()
    # data_end2 (data_end1 + 1年)
    d2_date = d1_date.replace(year=d1_date.year + 1)
    return d1_date.strftime("%Y%m%d"), d2_date.strftime("%Y%m%d")

# 获取今日日期
today = datetime.now().strftime("%Y%m%d")

# 计算相关日期
d1, d2 = get_financial_dates(today)

# 全局变量声明
global dates_1d, dates_1q, trade_day, stk_remain_df, info_dfs

dates_1d = pd.date_range(start=data_start, end=today).date
dates_1d = [d.strftime('%Y%m%d') for d in dates_1d]
dates_1q = pd.date_range(start=data_start, end=today, freq='q').date
dates_1q = [d.strftime('%Y%m%d') for d in dates_1q]

# 获取并保存交易日历
trade_day = pro.query('trade_cal', start_date='20050101', end_date=today)
trade_day = trade_day[trade_day['is_open'] == 1]['cal_date']
trade_day = trade_day.sort_values()
trade_day.to_csv('trade_date.csv', index=False)


#%% 数据下载与更新模块
def download_latest_data(today, data_end1):
    global stk_remain_df, close_1d, info_dfs, hs300, zz500
    
    # --------------------------
    # 1. 股票池清洗 (剔除ST、新股、停牌)
    # --------------------------
    if os.path.exists('stock_remain.csv'):
        stk_remain_df = pd.read_csv('stock_remain.csv', index_col=0)
        stk_remain_df.index = stk_remain_df.index.astype(str)
    else:
        stk_remain_df = pd.DataFrame()

    stk_all = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,name,list_date')
    stk_all = stk_all[~stk_all.ts_code.str.contains('BJ')].set_index('ts_code')  # 删除京股
    
    print("剔除新股、ST股、停牌股ing：")
    # 增量更新股票池
    for d in tqdm(dates_1d):
        if stk_remain_df.empty or d > stk_remain_df.index[-1]:
            try:
                stk_remain = stk_all[~stk_all.name.str.contains("ST")]  # 删除ST
                stk_remain = stk_remain[stk_remain.list_date < d]       # 删除上市不满1年的新股
                susp = pro.suspend_d(suspend_type='S', trade_date=d).set_index('ts_code') # 删除停牌
                stk_remain = pd.Series(1, index=(set(stk_remain.index) - set(susp.index)))
                stk_remain_df.loc[d] = stk_remain
            except:
                pass
            stk_remain_df.to_csv('stock_remain.csv')

    # --------------------------
    # 2. 更新每日收盘价 (后复权)
    # --------------------------
    if os.path.exists('close_1d.parquet'):
        close_1d = pd.read_parquet('close_1d.parquet')
        close_1d.index = close_1d.index.astype(str)
        existing_start = close_1d.index[0]
        if existing_start > data_start:
             print(f"警告：现有数据起点 {existing_start} 晚于要求的 {data_start}。建议重跑。")
    else:
        close_1d = pd.DataFrame()

    last_close_date = close_1d.index[-1] if not close_1d.empty else data_start
    
    if close_1d.empty:
        start_fetch_date = data_start
    else:
        start_fetch_date = (pd.to_datetime(last_close_date) + timedelta(days=1)).strftime('%Y%m%d')

    if today >= start_fetch_date:
        dates_to_update = [d.strftime('%Y%m%d') for d in pd.date_range(start=start_fetch_date, end=today)]
        
        if dates_to_update:
            print(f"需要更新行情数据: {start_fetch_date} -> {today} (共 {len(dates_to_update)} 天)")
            new_daily_data = []
            for d in tqdm(dates_to_update, desc="每日行情"):
                try:
                    df_daily = pro.daily(trade_date=d, fields='ts_code,close')
                    df_adj = pro.adj_factor(trade_date=d, fields='ts_code,adj_factor')
                    
                    if not df_daily.empty and not df_adj.empty:
                        df_merge = pd.merge(df_daily, df_adj, on='ts_code', how='inner')
                        df_merge['close_hfq'] = df_merge['close'] * df_merge['adj_factor']
                        df_merge['trade_date'] = d
                        new_daily_data.append(df_merge[['trade_date', 'ts_code', 'close_hfq']])
                except Exception as e:
                    print(f"{d} error: {e}")
            
            if new_daily_data:
                df_all_new = pd.concat(new_daily_data)
                new_closes_matrix = df_all_new.pivot(index='trade_date', columns='ts_code', values='close_hfq')
                new_closes_matrix.index = new_closes_matrix.index.astype(str)
                close_1d = pd.concat([close_1d, new_closes_matrix], axis=0).sort_index()
                close_1d.to_parquet('close_1d.parquet')

    # --------------------------
    # 3. 更新指数数据
    # --------------------------
    hs300 = pro.index_daily(ts_code='399300.SZ', start_date='20100101', end_date=today)
    hs300.index = hs300['trade_date'].astype(str)
    hs300.sort_index().to_csv('hs300.csv')
    
    zz500 = pro.index_daily(ts_code='000905.SH', start_date='20100101', end_date=today)
    zz500.index = zz500['trade_date'].astype(str)
    zz500.sort_index().to_csv('zz500.csv')

    # --------------------------
    # 4. 更新财报与估值指标
    # --------------------------
    info_dfs = {}
    conditions = ['roic','debt_to_assets','debt_to_eqt','ebit','roa','roe','fcff_ps','invest_capital','close','total_mv','pe','pb','ps','dv_ratio']
    
    last_fina_date = None
    
    # 读取现有数据
    for i in conditions:
        if os.path.exists(i+'.csv'):
            df = pd.read_csv(i+'.csv', index_col=0)
            df.index = df.index.astype(str)
            info_dfs[i] = df
            if i == 'roic' and not df.empty:
                last_fina_date = df.index[-1]
        else:
            info_dfs[i] = pd.DataFrame()

    # 计算缺失季度
    start_check = last_fina_date if last_fina_date else '20180101'
    
    if pd.to_datetime(data_end1) > pd.to_datetime(start_check):
        try:
            dates = pd.date_range(start=start_check, end=data_end1, freq='QE')
        except:
            dates = pd.date_range(start=start_check, end=data_end1, freq='Q')
            
        missing_quarters = [d.strftime('%Y%m%d') for d in dates if d.strftime('%Y%m%d') > start_check]
        
        if missing_quarters:
            print(f"开始补全财报数据: {missing_quarters}")
            all_codes = list(stk_remain_df.columns) if not stk_remain_df.empty else list(stk_all.index)

            for q_date in missing_quarters:
                chunk_size = 500
                df_fina_list = []
                for i in tqdm(range(0, len(all_codes), chunk_size), desc=f"下载 {q_date}"):
                    chunk = ",".join(all_codes[i:i+chunk_size])
                    try:
                        sub = pro.fina_indicator(ts_code=chunk, period=q_date, fields='ts_code,roic,debt_to_assets,debt_to_eqt,ebit,roa,roe,fcff_ps,invest_capital')
                        df_fina_list.append(sub)
                    except:
                        pass
                    time.sleep(0.1)
                
                if df_fina_list:
                    df_fina = pd.concat(df_fina_list).drop_duplicates(subset=['ts_code']).set_index('ts_code')
                    
                    # 补充交易日估值数据
                    search_d = q_date
                    df_daily = pd.DataFrame()
                    for _ in range(10):
                        try:
                            df_daily = pro.daily_basic(trade_date=search_d, fields='ts_code,close,total_mv,pe,pb,ps,dv_ratio')
                            if not df_daily.empty: break
                        except: pass
                        search_d = (pd.to_datetime(search_d) - timedelta(days=1)).strftime('%Y%m%d')
                    
                    if not df_daily.empty:
                        df_daily = df_daily.drop_duplicates(subset=['ts_code']).set_index('ts_code')
                        info_batch = pd.concat([df_fina, df_daily], axis=1)
                        
                        # 数据入库
                        for col in conditions:
                            if col in info_batch.columns:
                                if q_date not in info_dfs[col].index:
                                    info_dfs[col].loc[q_date] = np.nan
                                valid = info_batch.index.intersection(info_dfs[col].columns)
                                info_dfs[col].loc[q_date, valid] = info_batch.loc[valid, col]
                                info_dfs[col].to_csv(f'{col}.csv')

#%% 策略参数与选股逻辑
class Param:
    """
    策略参数配置类
    """
    def __init__(self, total_mv_high=600000, debt_to_eqt_high=0.5, roa_low=0.04, 
                 roic_low=0.04, pb_high=3, pe_high=25, ps_high=4, close_high=10, 
                 dv_ratio_low=1, lmt=20):
        self.total_mv_high = total_mv_high
        self.debt_to_eqt_high = debt_to_eqt_high
        self.roa_low = roa_low
        self.roic_low = roic_low
        self.pb_high = pb_high
        self.pe_high = pe_high
        self.ps_high = ps_high
        self.close_high = close_high
        self.dv_ratio_low = dv_ratio_low
        self.lmt = int(lmt)

    def get_dict(self):
        return {'total_mv_high': self.total_mv_high, 
                'debt_to_eqt_high': self.debt_to_eqt_high,
                'roa_low': self.roa_low,
                'roic_low': self.roic_low, 
                'pb_high': self.pb_high,
                'pe_high': self.pe_high,
                'ps_high': self.ps_high, 
                'close_high': self.close_high,
                'dv_ratio_low': self.dv_ratio_low,
                'lmt': self.lmt
               }

def select_stocks(start_date, end_date, method, param=Param()):
    """
    method: '华创' 或 '增强'
    根据策略选择符合条件的股票，返回dataframe，index为调仓日，values为符合条件的股票ID
    """
    # --- 内部因子计算函数 ---
    def total_mv(mv_df, low, high, bench_market=True):    
        mv_df = mv_df.reindex(dates_1q)
        if bench_market:
            market_mean = mv_df.mean(axis=1)
            low = market_mean + low
            high = market_mean + high
        bool_df = (mv_df.gt(low, axis=0) * mv_df.lt(high, axis=0))
        rank_df = pd.DataFrame(np.where(bool_df, mv_df, np.nan),
                             index=bool_df.index,
                             columns=bool_df.columns).rank(axis=1, ascending=False, method='dense')
        rank_df = rank_df.div(rank_df.max(axis=1), axis=0)
        return rank_df

    def debt_to_eqt(d2e_df, low, high, bench_market=True):
        d2e_df = d2e_df.reindex(dates_1q)
        if bench_market:
            market_mean = d2e_df.mean(axis=1)
            low = market_mean + low
            high = market_mean + high
        bool_df = (d2e_df.gt(low, axis=0) * d2e_df.lt(high, axis=0))
        rank_df = pd.DataFrame(np.where(bool_df, d2e_df, np.nan),
                             index=bool_df.index,
                             columns=bool_df.columns).rank(axis=1, ascending=False, method='dense')
        rank_df = rank_df.div(rank_df.max(axis=1), axis=0)
        return rank_df

    def fcff_ps(fcff_ps_df, low, high, bench_market=True):
        fcff_ps_df = fcff_ps_df.reindex(dates_1q)
        if bench_market:
            market_mean = fcff_ps_df.mean(axis=1)
            low = market_mean + low
            high = market_mean + high
        bool_df = (fcff_ps_df.gt(low, axis=0) * fcff_ps_df.lt(high, axis=0))
        rank_df = pd.DataFrame(np.where(bool_df, fcff_ps_df, np.nan),
                             index=bool_df.index,
                             columns=bool_df.columns).rank(axis=1, ascending=True, method='dense')
        rank_df = rank_df.div(rank_df.max(axis=1), axis=0)
        return rank_df

    def roa(roa_df, low, high, window_q=0, bench_market=True):
        roa_df = roa_df.reindex(dates_1q)
        if window_q > 0:
            roa_df = roa_df.rolling(window_q).mean()
        if bench_market:
            market_mean = roa_df.mean(axis=1)
            low = market_mean + low
            high = market_mean + high
        bool_df = (roa_df.gt(low, axis=0) * roa_df.lt(high, axis=0))
        rank_df = pd.DataFrame(np.where(bool_df, roa_df, np.nan),
                             index=bool_df.index,
                             columns=bool_df.columns).rank(axis=1, ascending=True, method='dense')
        rank_df = rank_df.div(rank_df.max(axis=1), axis=0)
        return rank_df

    def roic(roic_df, low, high, window_q=0, bench_market=True):
        roic_df = roic_df.reindex(dates_1q)
        if window_q > 0:
            roic_df = roic_df.rolling(window_q).mean()
        if bench_market:
            market_mean = roic_df.mean(axis=1)
            low = market_mean + low
            high = market_mean + high
        bool_df = (roic_df.gt(low, axis=0) * roic_df.lt(high, axis=0))
        rank_df = pd.DataFrame(np.where(bool_df, roic_df, np.nan),
                             index=bool_df.index,
                             columns=bool_df.columns).rank(axis=1, ascending=True, method='dense')
        rank_df = rank_df.div(rank_df.max(axis=1), axis=0)
        return rank_df

    def pe(pe_df, low, high, window_q=0, bench_market=True):
        pe_df = pe_df.reindex(dates_1q)
        if window_q > 0:
            pe_df = pe_df.rolling(window_q).mean()
        if bench_market:
            market_mean = pe_df.mean(axis=1)
            low = market_mean + low
            high = market_mean + high
        bool_df = (pe_df.gt(low, axis=0) * pe_df.lt(high, axis=0))
        rank_df = pd.DataFrame(np.where(bool_df, pe_df, np.nan),
                             index=bool_df.index,
                             columns=bool_df.columns).rank(axis=1, ascending=False, method='dense')
        rank_df = rank_df.div(rank_df.max(axis=1), axis=0)
        return rank_df

    def pb(pb_df, low, high, window_q=0, bench_market=True):
        pb_df = pb_df.reindex(dates_1q)
        if window_q > 0:
            pb_df = pb_df.rolling(window_q).mean()
        if bench_market:
            market_mean = pb_df.mean(axis=1)
            low = market_mean + low
            high = market_mean + high
        bool_df = (pb_df.gt(low, axis=0) * pb_df.lt(high, axis=0))
        rank_df = pd.DataFrame(np.where(bool_df, pb_df, np.nan),
                             index=bool_df.index,
                             columns=bool_df.columns).rank(axis=1, ascending=False, method='dense')
        rank_df = rank_df.div(rank_df.max(axis=1), axis=0)
        return rank_df

    def ps(ps_df, low, high, window_q=0, bench_market=True):
        ps_df = ps_df.reindex(dates_1q)
        if window_q > 0:
            ps_df = ps_df.rolling(window_q).mean()
        if bench_market:
            market_mean = ps_df.mean(axis=1)
            low = market_mean + low
            high = market_mean + high
        bool_df = (ps_df.gt(low, axis=0) * ps_df.lt(high, axis=0))
        rank_df = pd.DataFrame(np.where(bool_df, ps_df, np.nan),
                             index=bool_df.index,
                             columns=bool_df.columns).rank(axis=1, ascending=False, method='dense')
        rank_df = rank_df.div(rank_df.max(axis=1), axis=0)
        return rank_df

    def dv_ratio(dv_df, low, high, window_q=0, bench_market=True):
        dv_df = dv_df.reindex(dates_1q)
        if window_q > 0:
            dv_df = dv_df.rolling(window_q).mean()
        if bench_market:
            market_mean = dv_df.mean(axis=1)
            low = market_mean + low
            high = market_mean + high
        bool_df = (dv_df.gt(low, axis=0) * dv_df.lt(high, axis=0))
        rank_df = pd.DataFrame(np.where(bool_df, dv_df, np.nan),
                             index=bool_df.index,
                             columns=bool_df.columns).rank(axis=1, ascending=True, method='dense')
        rank_df = rank_df.div(rank_df.max(axis=1), axis=0)
        return rank_df

    def close(close_df, low, high, window_q=0, bench_market=False):
        close_df = close_df.reindex(dates_1q)
        if window_q > 0:
            close_df = close_df.rolling(window_q).mean()
        if bench_market:
            market_mean = close_df.mean(axis=1)
            low = market_mean + low
            high = market_mean + high
        bool_df = (close_df.gt(low, axis=0) * close_df.lt(high, axis=0))
        return bool_df
    
    # --- 策略分支 ---
    def select_by_hc(start_date,end_date, total_mv_high, debt_to_eqt_high, roa_low, roic_low,
             pb_high, pe_high, ps_high, close_high, dv_ratio_low, lmt):
        dates = pd.date_range(start=start_date, end=end_date, freq='1q').date
        dates = [d.strftime('%Y%m%d') for d in dates]

        total_mv_df = total_mv(info_dfs['total_mv'], low=-info_dfs['total_mv'].max().max(), high=0).reindex(dates)
        d2e_df = debt_to_eqt(info_dfs['debt_to_eqt'], -1000, 0).reindex(dates)
        fcff_ps_df = fcff_ps(info_dfs['fcff_ps'], 0, 1000).reindex(dates)
        roa_df = roa(info_dfs['roa'], 0, 1000, 12).reindex(dates)
        roic_df = roic(info_dfs['roic'], 0, 1000, 12).reindex(dates)
        pe_df = pe(info_dfs['pe'], -1000, 0, 4).reindex(dates)
        pb_df = pb(info_dfs['pb'], -1000, 0).reindex(dates)
        ps_df = ps(info_dfs['ps'], -1000, 0, 4).reindex(dates)

        scores_df = stk_remain_df.loc[dates,:] * (total_mv_df + d2e_df + fcff_ps_df + roa_df + roic_df + pe_df + pb_df + ps_df)
        return scores_df
    
    def select_by_strg(start_date, end_date, total_mv_high, debt_to_eqt_high, roa_low, roic_low,
             pb_high, pe_high, ps_high, close_high, dv_ratio_low, lmt):
        dates = pd.date_range(start=start_date, end=end_date, freq='1q').date
        dates = [d.strftime('%Y%m%d') for d in dates]
           
        total_mv_df = total_mv(info_dfs['total_mv'], low=0, high=total_mv_high, bench_market=False).reindex(dates)
        d2e_df = debt_to_eqt(info_dfs['debt_to_eqt'], -1000, debt_to_eqt_high, bench_market=False).reindex(dates)
        roa_df = roa(info_dfs['roa'], roa_low, 1000, bench_market=False).reindex(dates)
        roic_df = roic(info_dfs['roic'], roic_low, 1000, bench_market=False).reindex(dates)
        pe_df = pe(info_dfs['pe'], -1000, pe_high, bench_market=False).reindex(dates)
        pb_df = pb(info_dfs['pb'], -1000, pb_high, bench_market=False).reindex(dates)
        ps_df = ps(info_dfs['ps'], -1000, ps_high, bench_market=False).reindex(dates)
        dv_df = dv_ratio(info_dfs['dv_ratio'], dv_ratio_low, 10000, bench_market=False).reindex(dates)
        close_df = close(info_dfs['close'], 0, close_high, bench_market=False).reindex(dates)

        scores_df = stk_remain_df.loc[dates,:] * (total_mv_df + d2e_df + roa_df + roic_df + pe_df + pb_df + ps_df + dv_df) * close_df
        return scores_df
    
    # 策略路由
    if method == '华创':
        scores_df = select_by_hc(start_date, end_date, **param.get_dict())
    elif method == '增强':
        scores_df = select_by_strg(start_date, end_date, **param.get_dict())
    
    # 选出排名前 lmt 的股票
    rank_df = scores_df.rank(axis=1, method='first')
    rank_bool_df = rank_df.le(param.lmt, axis=0)
    
    stock_buy_list = [rank_bool_df.columns[rank_bool_df.iloc[i]].tolist() for i in range(len(rank_bool_df))]
    return pd.DataFrame(stock_buy_list, index=rank_bool_df.index)

#%% 回测执行引擎

def run(initial_capital, start_date, end_date, data_start, data_end2, method='增强', per_trade=0.05, param=Param()):
    """
    回测主函数
    """
    start_datetime = datetime.strptime(start_date, '%Y%m%d')
    end_datetime = datetime.strptime(end_date, '%Y%m%d')

    class Portfolio:
        def __init__(self, initial_capital, start_date, end_date, per_trade=per_trade):
            self.initial_capital = initial_capital
            self.capital = {start_date: self.initial_capital}
            self.cash = self.initial_capital
            self.stock_hold = {}  # {stock_id: [stock_num]}
            self.stock_hold_dict = {start_date: []}
            self.stock_hold_df = pd.DataFrame()
            self.stock_value = {start_date: 0}
            self.positions = pd.DataFrame()
            self.start_date = start_date
            self.end_date = end_date
            self.per_trade = per_trade
            self.number = 0
            self.close_tax = 0.0005
            self.open_commission = 0.0003
            self.close_commission = 0.0003
            self.rebalance_day_number = 0
            self.nv = []
            self.ret = pd.DataFrame()

        def sell(self, today, stock, amount, price):
            if amount == 0: return
            fee = (self.close_tax + self.close_commission) * amount * price
            self.cash = self.cash - fee + amount * price
            
            row_data = [today, '卖', stock, amount, price, fee]
            
            self.stock_hold[stock][0] -= amount 
            self.number += 1
            if self.stock_hold[stock][0] == 0:
                self.stock_hold.pop(stock)
            return row_data
                
        def buy(self, today, stock, amount, price):
            if amount == 0: return
            fee = self.open_commission * amount * price
            self.cash = self.cash - fee - amount * price
            
            if stock not in self.stock_hold.keys():
                self.stock_hold[stock] = [0]
            
            row_data = [today, '买', stock, amount, price, fee]
            self.stock_hold[stock][0] += amount
            self.number += 1
            return row_data
        
    def calculate_index(nv_series):
        """计算策略评价指标"""
        if len(nv_series) < 2:
            return {'年化收益率%':0, '回撤%':0, '年化波动率%':0, '夏普':0}
            
        ret_daily = nv_series.pct_change().fillna(0)
        days = len(nv_series)
        years_frac = days / 252.0 if days > 0 else 1.0

        total_ret = nv_series.iloc[-1] / nv_series.iloc[0] - 1
        if total_ret > -1:
            ann_return = ((1 + total_ret) ** (1 / years_frac) - 1) * 100
        else:
            ann_return = -100
            
        cummax = nv_series.cummax()
        drawdown = ((nv_series - cummax) / cummax).min()
        drawdown = abs(drawdown) * 100
        
        ann_vol = (ret_daily.std()) * math.sqrt(252) * 100
        sharp_ratio = (ann_return - 2.5) / ann_vol if ann_vol != 0 else 0

        index_dict = {
            '收益率%': total_ret * 100,
            '年化收益率%': ann_return, 
            '最大回撤%': drawdown, 
            '年化波动率%': ann_vol, 
            '夏普比率': sharp_ratio
        }
        return pd.Series(index_dict)
    
    portfolio = Portfolio(initial_capital, start_date, end_date)
    # 选股
    stocks_buy_df = select_stocks(data_start, end_date, method=method, param=param)
    
    # 交易日处理
    trade_start_end = pd.date_range(start=start_date, end=end_date, freq='d').date
    trade_start_end = [d.strftime('%Y%m%d') for d in trade_start_end]
    global trade_day  
    trade_day = sorted(set(trade_day).intersection(set(trade_start_end)))

    # 季度调仓日期
    rebalance_day = pd.date_range(start=data_start, end=data_end2, freq='1q').date
    rebalance_day = [d.strftime('%Y%m%d') for d in rebalance_day]
    cnt = 0

    for d in range(1, len(trade_day)):
        today = trade_day[d]
        preday = trade_day[d-1]
        
        trans = 0
        transactions = {}
        
        # 调仓逻辑：当前日期超过了预设的调仓日
        if today > rebalance_day[cnt+2]:
            stock_buy = list(stocks_buy_df.loc[rebalance_day[cnt]].dropna())
            
            # 计算总资产 (以上一日收盘价计算)
            stock_hold = list(portfolio.stock_hold.keys())
            adjust_price = np.array(close_1d.loc[preday, stock_hold])
            buy_amount = np.array([portfolio.stock_hold[key][0] for key in stock_hold])
            all_capital = portfolio.cash + np.sum(adjust_price * buy_amount)

            if len(stock_buy) > 0:
                per_cash = all_capital / len(stock_buy) 
            else:
                per_cash = 0
            
            # 平仓 (不在买入名单中的持仓)
            stock_hold_not_buy = list(set(stock_hold) - set(stock_buy))
            for stock in stock_hold_not_buy:
                price = close_1d.loc[preday, stock] 
                amount = portfolio.stock_hold[stock][0]
                trans += 1
                transactions[trans] = portfolio.sell(today, stock, amount, price)
                
            # 建仓 (在买入名单但不在持仓中)
            stock_buy_not_hold = list(set(stock_buy) - set(stock_hold))
            for stock in stock_buy_not_hold:
                price = close_1d.loc[preday, stock]
                amount = int(per_cash / price)
                trans += 1
                transactions[trans] = portfolio.buy(today, stock, amount, price)
                
            # 调仓 (交集部分)
            stock_buy_hold = list(set(stock_buy).intersection(set(stock_hold)))
            for stock in stock_buy_hold:
                price = close_1d.loc[preday, stock] 
                amount = int(per_cash / price) - portfolio.stock_hold[stock][0]
                if amount < 0:
                    trans += 1
                    transactions[trans] = portfolio.sell(today, stock, abs(amount), price)
                else:
                    trans += 1
                    transactions[trans] = portfolio.buy(today, stock, amount, price)
               
            portfolio.capital[today] = all_capital
            portfolio.stock_value = all_capital - portfolio.cash
            portfolio.stock_hold_dict[today] = list(portfolio.stock_hold.keys())
            cnt += 1 
            
            transactions = pd.DataFrame(transactions, index=['交易日','交易类型','股票代码','股票数量','价格','交易费用']).T.reset_index().set_index('交易日')
            transactions.columns = ['交易编号','交易类型','股票代码','股票数量','价格','交易费用']
            portfolio.positions = pd.concat([portfolio.positions, transactions], axis=0)
        else:
            # 非调仓日，更新资产价值
            portfolio.stock_hold_dict[today] = list(portfolio.stock_hold.keys())
            stock_value = 0
            for stock in portfolio.stock_hold_dict[today]:
                stock_value += portfolio.stock_hold[stock][0] * close_1d.loc[preday, stock]
                
            portfolio.stock_value = stock_value
            portfolio.capital[today] = portfolio.cash + portfolio.stock_value
            
    # 持仓与交易打印
    print("{} 的持仓记录：\n".format(today), portfolio.stock_hold_dict[today])
    if trans == 0:
        print('{}操作：无交易'.format(today))
    else: 
        print("{} 交易记录：\n".format(today), transactions)
            
    # 生成结果
    stock_hold_df = pd.DataFrame({key: pd.Series(value, dtype='str') for key, value in portfolio.stock_hold_dict.items()}).T
    stock_hold_df.columns = range(1, len(stock_hold_df.columns) + 1)
    stock_hold_df = stock_hold_df.fillna('')
    portfolio.stock_hold_df = stock_hold_df
    
    # 净值计算
    portfolio.capital = pd.Series(portfolio.capital)
    portfolio.nv = (portfolio.capital.iloc[1:] / initial_capital).reindex(trade_day).fillna(method='ffill').fillna(1.0)
    portfolio.nv.name = '净值'
    portfolio.ret = portfolio.nv.pct_change().fillna(0)
    
    strategy_index = calculate_index(portfolio.nv)
    result_index = pd.DataFrame([strategy_index], index=['策略'])
    
    # 沪深300对比
    global hs300
    hs300 = pd.read_csv(r'hs300.csv')
    hs300.index = hs300['trade_date'].astype(str)
    hs300 = hs300.loc[trade_day, 'close']
    hs300 = hs300.pct_change().fillna(0)
    hs300 = (1 + hs300).cumprod().dropna()
    hs300_index = calculate_index(hs300)
    result_index = pd.concat([result_index, pd.DataFrame([hs300_index], index=['沪深300'])], axis=0)
    
    # 中证500对比
    global zz500
    zz500 = pd.read_csv(r'zz500.csv')
    zz500.index = zz500['trade_date'].astype(str)
    zz500 = zz500.loc[trade_day, 'close']
    zz500 = zz500.pct_change().fillna(0)
    zz500.fillna(0, inplace=True)
    zz500 = (1 + zz500).cumprod().dropna()
    zz500_index = calculate_index(zz500)
    result_index = pd.concat([result_index, pd.DataFrame([zz500_index], index=['中证500'])], axis=0)
    
    print(result_index)
    return portfolio, result_index, portfolio.nv


def generate_daily_report(portfolio, close_1d, hs300_full, zz500_full):
    """
    生成符合要求的每日详细监控报告
    """
    import math
    
    def calc_metrics_internal(nv_series):
        if len(nv_series) < 2:
            return pd.Series({'收益率%':0, '年化收益率%':0, '最大回撤%':0, '年化波动率%':0, '夏普比率':0})
        
        days = len(nv_series)
        years_frac = days / 252.0 if days > 0 else 1.0
        total_ret = nv_series.iloc[-1] / nv_series.iloc[0] - 1
        
        if total_ret > -1:
            ann_return = ((1 + total_ret) ** (1 / years_frac) - 1) * 100
        else:
            ann_return = -100
            
        cummax = nv_series.cummax()
        drawdown = ((nv_series - cummax) / cummax).min()
        drawdown = abs(drawdown) * 100
        
        ret_daily = nv_series.pct_change().fillna(0)
        ann_vol = (ret_daily.std()) * math.sqrt(252) * 100
        sharp_ratio = (ann_return - 2.5) / ann_vol if ann_vol != 0 else 0

        return pd.Series({
            '收益率%': total_ret * 100,
            '年化收益率%': ann_return, 
            '最大回撤%': drawdown, 
            '年化波动率%': ann_vol, 
            '夏普比率': sharp_ratio
        })

    def plot_period_internal(data_slice, bench_slice, title, filename):
        if data_slice.empty: return
        strategy_norm = data_slice / data_slice.iloc[0]
        bench_norm = bench_slice / bench_slice.iloc[0]
        
        plt.figure(figsize=(12, 5))
        plt.plot(strategy_norm, label='策略', color='red', linewidth=2)
        plt.plot(bench_norm, label='沪深300', color='gray', linestyle='--', alpha=0.7)
        plt.title(f"{title}净值走势 ({data_slice.index[0]} - {data_slice.index[-1]})", fontsize=15)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    print("\n" + "="*60)
    print("                策略每日监控报告")
    print("="*60)

    nv = portfolio.nv
    today_str = nv.index[-1]
    prev_day_str = nv.index[-2] if len(nv) > 1 else today_str
    
    bench_hs300 = hs300_full.reindex(nv.index).fillna(method='ffill')
    
    print(f"\n【1 & 2. 持仓信息与今日表现】 (日期: {today_str})")
    print(f"策略今日净值: {nv.iloc[-1]:.4f} | 今日涨跌幅: {(nv.iloc[-1]/nv.iloc[-2]-1)*100:.2f}%")
    
    current_holdings = portfolio.stock_hold_dict.get(today_str, [])
    
    if not current_holdings:
        print("当前无持仓。")
    else:
        hold_data = []
        total_asset_value = portfolio.capital[today_str]
        
        for code in current_holdings:
            amount = portfolio.stock_hold.get(code, [0])[0]
            price_today = close_1d.loc[today_str, code] if code in close_1d.columns else 0
            price_prev = close_1d.loc[prev_day_str, code] if code in close_1d.columns else price_today
            
            mv = amount * price_today
            weight = (mv / total_asset_value) * 100
            daily_ret = (price_today / price_prev - 1) * 100 if price_prev > 0 else 0
            
            hold_data.append({
                '代码': code,
                '持仓数量': amount,
                '仓位权重(%)': round(weight, 2),
                '今日涨跌(%)': round(daily_ret, 2),
                '当前价格': price_today,
                '持仓市值': round(mv, 2)
            })
            
        df_hold = pd.DataFrame(hold_data)
        print("\n>>> 持仓明细 (Open -> Close):")
        print(df_hold[['代码', '仓位权重(%)', '今日涨跌(%)', '当前价格', '持仓市值']].to_string(index=False))
        print("-" * 50)

    print(f"\n【3. 过去一个月表现 (近22交易日)】")
    if len(nv) >= 22:
        nv_1m = nv.iloc[-22:]
        bench_1m = bench_hs300.iloc[-22:]
        metrics_1m = calc_metrics_internal(nv_1m)
        print(metrics_1m.to_frame(name='近1月指标').T.to_string())
        plot_period_internal(nv_1m, bench_1m, '近一个月', 'report_1month.png')
        print(">> 图表已保存: report_1month.png")
    else:
        print("数据不足22天，跳过月度统计。")

    print(f"\n【4. 过去一年表现 (近252交易日)】")
    if len(nv) >= 252:
        nv_1y = nv.iloc[-252:]
        bench_1y = bench_hs300.iloc[-252:]
        metrics_1y = calc_metrics_internal(nv_1y)
        print(metrics_1y.to_frame(name='近1年指标').T.to_string())
        plot_period_internal(nv_1y, bench_1y, '近一年', 'report_1year.png')
        print(">> 图表已保存: report_1year.png")
    else:
        print("数据不足252天，跳过年度统计。")

    print(f"\n【5. 策略完整历史表现】")
    full_metrics = calc_metrics_internal(nv)
    print(full_metrics.to_frame(name='全历史指标').T.to_string())

#%% 绘制图像
def nv_plot(today, result_index):
    plt.figure(figsize=(15,6))
    plt.subplot()
    plt.plot(portfolio.nv.loc[trade_day[1]:], label='策略')
    plt.plot(hs300.loc[trade_day[1]:]/hs300.loc[trade_day[1]], label='沪深300指数')
    plt.plot(zz500.loc[trade_day[1]:]/zz500.loc[trade_day[1]], label='中证500指数')
    plt.xticks(ticks=portfolio.nv.loc[trade_day[1]:].index[0::int(len(portfolio.nv.loc[trade_day[1]:])/10)])
    plt.grid()  
    plt.title('{}净值曲线'.format(today), fontsize=20)
    plt.legend(loc='upper left')
    plt.text(x=0.8, y=0.9,
             s="年化收益：{:.4f}%".format(result_index.loc['策略','年化收益率%']),
             fontsize=18,
             transform=plt.gca().transAxes)
    plt.savefig(r'截至今日净值图.png')
    print("---------------净值图已保存----------------")
def generate_html_report(portfolio, close_1d, today_str):
    import math
    import base64
    import os
    print("正在生成惠特尼乔治策略监控报告...")
    
    today_ret = portfolio.ret.iloc[-1] if not portfolio.ret.empty else 0.0
    
    # --- 保留你的修复逻辑: 智能判断日期 ---
    if today_str in close_1d.index:
        current_date_use = today_str
    else:
        current_date_use = close_1d.index[-1]
        print(f"警告: 未找到日期 {today_str} 的行情数据，使用 {current_date_use}。")

    # --- 1. 核心改动：判定调仓建议 (基于你原版 portfolio.positions 记录) ---
    is_rebalance_day = False
    suggestion_section = ""
    
    # 判断今天是否是决策日：检查 positions 最后一行索引是否是今天
    if not portfolio.positions.empty and str(portfolio.positions.index[-1]) == str(today_str):
        is_rebalance_day = True

    if is_rebalance_day:
        # 调仓日：展示具体动作
        last_pos = portfolio.positions.iloc[-1].dropna()
        # 这里提取你原版代码中记录的调仓明细
        trade_rows = ""
        # 假设你的 positions 记录了 '股票代码', '交易类型' 等列（根据你原文件逻辑）
        # 如果是简单的 DataFrame 格式：
        if '股票代码' in last_pos.index or isinstance(portfolio.positions.iloc[-1], pd.Series):
             # 提取当前调仓名单（简版展示）
             target_list = portfolio.positions.loc[today_str]
             if isinstance(target_list, pd.DataFrame):
                 for _, row in target_list.iterrows():
                     color = "red" if row['交易类型'] == '买' else "green"
                     trade_rows += f"<tr><td>{row['股票代码']}</td><td>{row['交易类型']}</td><td style='color:{color}; font-weight:bold;'>执行调仓</td></tr>"
        
        suggestion_section = f"""
        <div class="section">
            <h2 style="color: #e67e22;">下一交易日持仓建议 - [ 需调仓 ]</h2>
            <p>⚠️ <b>惠特尼乔治策略提示:</b> 检测到模型调仓信号，请执行以下操作：</p>
            <table>
                <tr><th>股票代码</th><th>方向</th><th>变化建议</th></tr>
                {trade_rows if trade_rows else "<tr><td colspan='3'>请参考下方持仓明细进行同步</td></tr>"}
            </table>
        </div>
        """
    else:
        # 非调仓日：标注不调仓
        suggestion_section = f"""
        <div class="section">
            <h2 style="color: #27ae60;">下一交易日持仓建议 - 无止损策略</h2>
            <p><span class="check-icon">✅ 不需要调仓:</span> 各资产权重均在目标范围内，维持当前持仓。</p>
            <table>
                <tr><th>资产类别</th><th>当前权重</th><th>建议权重</th><th>目标权重</th><th>变化</th></tr>
                <tr><td>股票 (惠特尼乔治)</td><td>100.0%</td><td>100.0%</td><td>100%</td><td>无变化</td></tr>
            </table>
        </div>
        """

    # --- 保留你的原始数据处理逻辑 ---
    df_return = pd.DataFrame({
        'Daily Return': [f"{today_ret:.6f}"],
        'Total NV': [f"{portfolio.nv.iloc[-1]:.4f}"]
    }, index=[today_str])
    
    holdings_data = []
    total_asset = portfolio.capital.iloc[-1] if hasattr(portfolio.capital, 'iloc') else portfolio.capital.get(today_str, 0)
    
    current_holdings_codes = []
    if hasattr(portfolio, 'stock_hold_dict'):
         current_holdings_codes = portfolio.stock_hold_dict.get(today_str, [])
    
    all_dates = sorted(close_1d.index.tolist())
    try:
        curr_idx = all_dates.index(current_date_use)
        prev_date = all_dates[curr_idx - 1] if curr_idx > 0 else current_date_use
    except ValueError:
        prev_date = current_date_use

    for code in current_holdings_codes:
        amount_list = portfolio.stock_hold.get(code)
        amount = amount_list[0] if amount_list else 0
        if amount > 0:
            price_now = close_1d.loc[current_date_use, code] if code in close_1d.columns else 0
            price_prev = close_1d.loc[prev_date, code] if code in close_1d.columns else price_now
            mkt_val = amount * price_now
            weight = mkt_val / total_asset if total_asset > 0 else 0
            stk_ret = (price_now / price_prev) - 1 if (price_prev and price_prev > 0) else 0
            holdings_data.append({'Symbol': code, 'Side': 'Long', 'Weight': weight, 'Return': stk_ret})
    
    df_holdings = pd.DataFrame(holdings_data)
    if not df_holdings.empty:
        df_holdings = df_holdings.sort_values(by='Weight', ascending=False)
        df_holdings['Weight'] = df_holdings['Weight'].apply(lambda x: f"{x:.2%}")
        df_holdings['Return'] = df_holdings['Return'].apply(lambda x: f"{x:.6f}")
    else:
        df_holdings = pd.DataFrame(columns=['Symbol', 'Side', 'Weight', 'Return'])

    # --- 保留你的 calculate_metrics 函数 ---
    def calculate_metrics(series):
        if len(series) < 2: return ['-', '-', '-', '-', '-', '-']
        ret_series = series.pct_change().fillna(0)
        days = len(series)
        tot_ret = series.iloc[-1] / series.iloc[0] - 1
        years_frac = max(days / 252.0, 0.01) 
        ann_ret = (1 + tot_ret) ** (1/years_frac) - 1 if tot_ret > -1 else -1.0
        ann_vol = ret_series.std() * math.sqrt(252)
        sharpe = (ann_ret - 0.025) / ann_vol if ann_vol != 0 else 0
        max_dd = ((series - series.cummax()) / series.cummax()).min()
        win_rate = ret_series[ret_series > 0].count() / ret_series[ret_series != 0].count() if ret_series[ret_series != 0].count() > 0 else 0
        return [f"{ann_ret:.2%}", f"{ann_vol:.2%}", f"{sharpe:.2f}", f"{max_dd:.2%}", "-", f"{win_rate:.2%}"]

    metric_names = ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio', 'Win Rate']
    metrics_1m = calculate_metrics(portfolio.nv.iloc[-22:])
    metrics_1y = calculate_metrics(portfolio.nv.iloc[-252:])
    metrics_all = calculate_metrics(portfolio.nv)

    df_metrics = pd.DataFrame({
        'Last Month (22D)': metrics_1m,
        'Last Year (252D)': metrics_1y,
        'Inception (All)': metrics_all
    }, index=metric_names)

    # --- 保留图片逻辑 ---
    img_path = '截至今日净值图.png'
    img_html = ""
    if os.path.exists(img_path):
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            img_html = f'<div style="text-align:center;"><img src="data:image/png;base64,{encoded_string}" style="max-width:90%; border:1px solid #ddd; padding:5px;"></div>'

    # --- 最终 HTML 拼接 (融合你要求的格式) ---
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>策略监控面板 - {today_str}</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f9; color: #333; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 0 15px rgba(0,0,0,0.1); border-radius: 8px; }}
            h1 {{ text-align: center; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 15px; }}
            h2 {{ color: #34495e; margin-top: 30px; border-left: 5px solid #3498db; padding-left: 10px; font-size: 1.2em; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 10px; font-size: 13px; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: right; }}
            th {{ background-color: #f8f9fa; color: #555; text-align: center; }}
            .check-icon {{ color: #27ae60; font-weight: bold; }}
            .footer {{ margin-top: 40px; text-align: center; font-size: 12px; color: #999; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>策略监控面板 - 报告日期: {today_str[:4]}-{today_str[4:6]}-{today_str[6:]}</h1>
            
            <h2>目标资产配置权重</h2>
            <p style="padding-left: 15px;">股票 (惠特尼乔治): 100%<br>
            调仓阈值: ±10% (偏离目标权重超过阈值或季度更新时调仓)</p>

            <h2>今日持仓情况</h2>
            {df_holdings.to_html(classes='dataframe', index=False, justify='center')}

            {suggestion_section}

            <h2>今日策略收益率</h2>
            {df_return.to_html(classes='dataframe', justify='center')}

            <h2>策略表现指标</h2>
            {df_metrics.to_html(classes='dataframe', justify='center')}
            
            <h2>净值走势图</h2>
            {img_html}
            
            <div class="footer">惠特尼乔治策略系统 | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
    </body>
    </html>
    """
    
    try:
        with open('results.html', "w", encoding="utf-8") as f:
            f.write(html_content)
        print("------------HTML 报告已按照新格式成功生成: results.html------------")
    except Exception as e:
        print(f"写入失败: {e}")
#%% 主程序执行
if __name__ == "__main__":
    begin_time = time.time()
    
    # 1. 更新数据
    download_latest_data(today=today, data_end1=d1)
    print('------------数据更新完成------------')

    # 2. 运行策略
    portfolio, performace, nv = run(initial_capital=1000000, 
                                    start_date=start_date, end_date=today,
                                    data_start=data_start, data_end2=d2,
                                    method='增强', per_trade=0.05)
    print('------------策略运行完成-------------')

    # 3. 准备基准数据并生成监控报告
    hs300_full = pd.read_csv('hs300.csv', dtype={'trade_date': str}).set_index('trade_date')['close']
    zz500_full = pd.read_csv('zz500.csv', dtype={'trade_date': str}).set_index('trade_date')['close']

    generate_daily_report(portfolio, close_1d, hs300_full, zz500_full)

    # 4. 保存结果与生成 HTML 报告
    nv_plot(today, performace)
    portfolio.positions.to_excel('交易记录.xlsx', index=False)
    portfolio.stock_hold_df.to_excel('当前持股.xlsx')
    portfolio.ret.to_excel('收益明细.xlsx')
    performace.to_excel('策略表现指标.xlsx')
    
    end_time = time.time()
    generate_html_report(portfolio, close_1d, today)

    print('============================================')
    print('总用时（包括数据更新）：{}'.format(time.strftime("%H 时 %M 分 %S 秒", time.gmtime(end_time - begin_time))))

    HTML_PATH = "/Users/guangdafei/PythonProjects/量化/崇文/惠特尼乔治2026/data/results.html"  # 如有问题可以改成absolute path
    # HTML_PATH = "data/results.html"
    try:
        with open(HTML_PATH, "r", encoding="utf-8") as f:
            HTML_BODY = f.read()
    except Exception:
        HTML_BODY = "<p>Please find the attached file.</p>"
    # HTML_BODY = "<p>Please find the attached file.</p>"
    for re in pw.RECIPIENTS:
        print(f"Sending {HTML_PATH} to {re}")
        email_sender_v2.send_html_email_with_attachment(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            sender_email=pw.SENDER_EMAIL,    # your gmail
            password=pw.google_email_app_password,  # your gmail app password
            receiver_email=re,  # recipient email
            subject="Whitney George Daily Backtest Report",
            html_body=HTML_BODY,
            attachment_path=HTML_PATH
        )