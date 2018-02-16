# -*- coding:utf8 -*- 
#! /usr/bin/python
#coding:utf-8
import sys,math
from odps.udf import annotate
from odps.udf import BaseUDTF
"""
item_id,title,cate_id,cate_level1_id,
shop_id,org_brand_id,seller_id,discrete_feats
"""
#udtf
#---------2
lst_online_time_bin=[1,7,14,30,60,180,365] # BIGINT COMMENT '商品上架时长',
se_lpv_pc_1d_001_bin=[2**x for x in range(0,7)] #BIGINT COMMENT '商品最近1天在搜索中的PV（pc主搜）',
se_ipv_pc_1d_001_bin=[2**x for x in range(0,10)] #BIGINT COMMENT '商品最近1天在搜索中的IPV（pc主搜）',
se_lpv_pc_1m_001_bin=[2**x for x in range(0,15)] # BIGINT COMMENT '商品30天搜索PV（pc主搜）',
expo_uv_1m_001_bin=[2**x for x in range(0,10)] #BIGINT COMMENT '商品30天搜索UV（pc主搜）',
clk_cnt_1m_015_bin=[2**x for x in range(0,8)] #BIGINT COMMENT '商品30天搜索IPV（pc主搜）',
clk_uv_1m_004_bin=[2**x for x in range(0,8)] #BIGINT COMMENT '商品30天搜索IUV（pc主搜）',
#----------3-----------
vst_cart_byr_rate_1m_001_bin=[0,0.01,0.04,0.06,0.1,0.16,0.2,0.5,0.6,0.8] #DOUBLE COMMENT '商品30天浏览加购率：30天加购UV/30天UV,(cart_byr_cnt_1m_001/ipv_uv_1m_002)',
vst_clt_byr_rate_1m_001_bin=vst_cart_byr_rate_1m_001_bin #DOUBLE COMMENT '商品30天浏览收藏率：30天收藏UV/30天UV,(clt_byr_cnt_1m_016/ipv_uv_1m_002)'
cart_cnt_3m_002_bin=[0,1,2,5,10,100,1024] #BIGINT COMMENT '商品90天日均加入购物车数',
#----------4-------------
vst_pay_byr_rate_1d_001_bin=vst_cart_byr_rate_1m_001_bin  #DOUBLE COMMENT '商品最近1天成交支付买家转化率(1天成交UV/全网UV),(pay_ord_byr_cnt_1d_001/ipv_uv_1d_001)',
pay_ord_byr_cnt_3d_001_bin=[2**x for x in range(0,10)]  #BIGINT COMMENT '商品最近3天成交支付买家数',
vst_pay_byr_rate_3d_001_bin=vst_cart_byr_rate_1m_001_bin #DOUBLE COMMENT '商品最近3天成交支付买家转化率(3天成交UV/全网UV),(pay_ord_byr_cnt_3d_001/ipv_uv_3d_001)',
vst_pay_byr_rate_1w_017_bin=vst_cart_byr_rate_1m_001_bin #DOUBLE COMMENT '商品最近7天成交支付买家转化率(7天成交UV/全网UV),(pay_ord_byr_cnt_1w_001/ipv_uv_1w_001)',
pay_ord_byr_cnt_2w_001_bin=[2**x for x in range(0,10)] #BIGINT COMMENT '商品最近15天成交支付买家数',
vst_pay_byr_rate_2w_001_bin=vst_cart_byr_rate_1m_001_bin #DOUBLE COMMENT '商品最近15天成交支付买家转化率(15天成交UV/全网UV),(pay_ord_byr_cnt_2w_001/ipv_uv_2w_001)',
vst_pay_byr_rate_1m_005_bin=vst_cart_byr_rate_1m_001_bin #DOUBLE COMMENT '商品最近30天成交支付买家转化率(30天成交UV/全网UV),(pay_ord_byr_cnt_1m_002/ipv_uv_1m_002)',
#----------5--------------
pay_ord_cnt_1d_015_bin=[2**x for x in range(0,10)] #BIGINT COMMENT '商品最近1天搜索引导成交笔数',
pay_ord_cnt_1m_043_bin=[2**x for x in range(0,14)] #BIGINT COMMENT '商品最近30天搜索引导成交笔数',
pay_ord_byr_cnt_1m_076_bin=[2**x for x in range(0,14)]  #BIGINT COMMENT '商品最近30天搜索引导成交UV',
vst_crt_byr_rate_1m_002_bin=vst_cart_byr_rate_1m_001_bin  #DOUBLE COMMENT '商品最近30天搜索引导成交转化率：搜索成交UV/搜索UV,(pay_ord_byr_cnt_1m_076/expo_uv_1m_001)',
vst_crt_byr_rate_1m_001_bin=vst_cart_byr_rate_1m_001_bin  #DOUBLE COMMENT '商品最近30天IPV成交转化率：成交UV/全网IPV,(pay_ord_byr_cnt_1m_002/ipv_1m_001)',
byr_rmk_rate_1m_003_bin=[0,0.5,0.8]  # DOUBLE COMMENT '商品30天差评率:(byr_rmk_cnt_1m_003/byr_rmk_cnt_1m_004)',
byr_rmk_rate_1m_002_bin=[0,0.5,0.9,0.97]  # DOUBLE COMMENT '商品30天好评率:(byr_rmk_cnt_1m_001/byr_rmk_cnt_1m_004)',
#-----------6--------------
ipv_uv_3d_001_bin=[2**x for x in range(0,20)] # BIGINT COMMENT '最近3天商品UV',
ipv_uv_2w_001_bin=[2**x for x in range(0,20)] # BIGINT COMMENT '最近15天商品UV',
ipv_1d_001_bin=[2**x for x in range(0,20)] #BIGINT COMMENT '最近1天IPV',
ipv_uv_1d_001_bin=[2**x for x in range(0,20)] #100w, BIGINT COMMENT '最近1天IPV_UV',
ipv_3m_003_bin=[2**x for x in range(0,15)] #15W, BIGINT COMMENT '最近90天日均ipv',
ipv_1w_001_bin=[2**x for x in range(0,20)] # BIGINT COMMENT '最近7天IPV',
ipv_1m_001_bin=[2**x for x in range(0,20)] #BIGINT COMMENT '最近30天IPV',
ipv_uv_1w_001_bin=[2**x for x in range(0,15)] #BIGINT COMMENT '最近7天商品UV',
ipv_uv_1m_002_bin=[2**x for x in range(0,20)] #BIGINT COMMENT '最近30天商品UV',
#-------7-------------
byr_rmk_cnt_6m_004_bin=[2**x for x in range(0,21)]   # --      BIGINT COMMENT '最近180天买家评价数'   #
byr_rmk_cnt_1m_004_bin=[2**x for x in range(0,20)]   # --      BIGINT COMMENT '最近30天买家评价数'   #
byr_rmk_cnt_1m_003_bin=[2**x for x in range(0,10)]   # --      BIGINT COMMENT '最近30天差评买家评价数'   #
byr_rmk_cnt_1m_001_bin=[2**x for x in range(0,15)]   # --      BIGINT COMMENT '最近30天好评买家评价数'   #
#------8-------------------
clt_cnt_3d_001_bin=[10**x-1 for x in range(0,5)]   #--           BIGINT COMMENT '最近3天商品收藏次数'   #
clt_cnt_1w_002_bin=[2**x for x in range(0,15)]   #--           BIGINT COMMENT '最近7天商品收藏次数'   #
clt_cnt_2w_001_bin=[2**x for x in range(0,15)]   #--           BIGINT COMMENT '最近14天商品收藏次数'   #
clt_cnt_1m_002_bin=[2**x for x in range(0,18)]   #--           BIGINT COMMENT '最近30天商品收藏次数'   #
clt_cnt_2m_003_bin=[2**x for x in range(0,20)]   #--           BIGINT COMMENT '最近60天商品收藏次数'   #
clt_cnt_3m_003_bin=[2**x for x in range(0,20)]   #--           BIGINT COMMENT '最近90天商品收藏次数'   #
clt_cnt_3m_004_bin=[10**x-1 for x in range(0,5)]   #--           BIGINT COMMENT '最近90天日均商品收藏次数'   #
clt_byr_cnt_1m_016_bin=[2**x for x in range(0,18)]   #--       BIGINT COMMENT '最近30天商品收藏买家数'   #
#-------9----------------
cart_cnt_3d_001_bin=[10**x-1 for x in range(0,5)]   #--          BIGINT COMMENT '最近3天加购物车次数'   #
cart_cnt_1w_001_bin=[2**x for x in range(0,20)]   #--          BIGINT COMMENT '最近7天加入购物车次数'   #
cart_cnt_2w_001_bin=[2**x for x in range(0,20)]   #--          BIGINT COMMENT '最近15天加购物车次数'   #
cart_cnt_1m_001_bin=[2**x for x in range(0,20)]   #--          BIGINT COMMENT '最近30天加入购物车次数'   #
cart_cnt_2m_001_bin=[2**x for x in range(0,20)]   #--          BIGINT COMMENT '最近60天加购物车次数'   #
cart_cnt_3m_001_bin=[2**x for x in range(0,20)]   #--          BIGINT COMMENT '最近90天加购物车次数'   #
cart_byr_cnt_1m_001_bin=[2**x for x in range(0,20)]   #--      BIGINT COMMENT '最近30天加入购物车的买家数'   #
#-------10--------------------------
pay_ord_byr_cnt_1d_001_bin=[2**x for x in range(0,10)]   #--   BIGINT COMMENT '最近1天支付买家数'   #
pay_ord_byr_cnt_1w_001_bin=[2**x for x in range(0,15)]   #--   BIGINT COMMENT '最近7天支付买家数'   #
pay_ord_cnt_3m_004_bin=[2**x for x in range(0,10)]   #--       BIGINT COMMENT '最近90天日均支付子订单数(日均成交笔数)'   #
pay_ord_byr_cnt_1m_002_bin=[2**x for x in range(0,20)]   #--   BIGINT COMMENT '最近30天支付买家数'   #
pay_ord_itm_qty_3m_001_bin=[2**x for x in range(0,20)]   #--   BIGINT COMMENT '最近90天支付商品件数(商品累计销量)'   #
apl_rfd_cnt_1m_001_bin=[2**x for x in range(0,8)]   #--       BIGINT COMMENT '最近30天申请退款笔数'   #
dspt_rfd_cnt_1m_001_bin=[2**x for x in range(0,8)]   #--      BIGINT COMMENT '最近30天纠纷退款笔数'   #
dspt_rfd_rate_1m_001_bin=vst_cart_byr_rate_1m_001_bin   #--     DOUBLE COMMENT '最近30天纠纷退款率:(dspt_rfd_cnt_1m_001/apl_rfd_cnt_1m_001)'   #
#---------11----
#is_baojianpin_bin=[]             #--  STRING COMMENT '保健品资质打标   #来源于dump的lasttable'   #
#is_diaopai_bin=[]                #--  STRING COMMENT '吊牌打标   #来源于dump的lasttable'   #
#is_fangxintao_bin=[]             #--  STRING COMMENT '放心淘打标   #来源于dump的lasttable'   #
#is_foodqs_bin=[]                 #--  STRING COMMENT '食品打标   #来源于dump的lasttable'   #
#is_guanfangzhishou=[]        #--  STRING COMMENT '官方直售打标   #来源于dump的lasttable'   #
#is_haiwaizhiyou           #--  STRING COMMENT '海外直邮打标   #来源于dump的lasttable'   #
#is_huazhuangpin           #--  STRING COMMENT '化妆品资质打标   #来源于dump的lasttable'   #
#is_huodaofukuan           #--  STRING COMMENT '货到付款打标   #来源于dump的lasttable'   #
#is_jiajubaihuo            #--  STRING COMMENT '家居百货打标   #来源于dump的lasttable'   #
#is_mianfeihuanxin         #--  STRING COMMENT '免费换新打标   #来源于dump的lasttable'   #
#is_pinpaishouquan         #--  STRING COMMENT '品牌授权打标   #来源于dump的lasttable'   #
#is_pinzhichengnuo         #--  STRING COMMENT '品质承诺打标   #来源于dump的lasttable'   #
#is_poshunbuji             #--  STRING COMMENT '破损补寄   #来源于dump的lasttable'   #
#is_qiyemaijia             #--  STRING COMMENT '企业卖家   #来源于dump的lasttable'   #
#is_quanqiugou             #--  STRING COMMENT '全球购   #来源于dump的lasttable'   #
#is_rzershouche            #--  STRING COMMENT '认证二手车   #来源于dump的lasttable'   #
#is_tianmaoguoji           #--  STRING COMMENT '天猫国际   #来源于dump的lasttable'   #
#is_tuihuoyunfei           #--  STRING COMMENT '退货运费   #来源于dump的lasttable'   #
#is_xinpin                 #--  STRING COMMENT '是否新品   #来源于dump的lasttable'   #
#is_yiyaozizhi             #--  STRING COMMENT '医药资质   #来源于dump的lasttable'   #
#is_zhichianzhuang         #--  STRING COMMENT '支持安装   #来源于dump的lasttable'   #
#is_zhongguozhizao         #--  STRING COMMENT '中国制造   #来源于dump的lasttable'   #
#is_baoyou                 #--  STRING COMMENT '是否包邮   #来源dim_tb_itm的shipping字段'   #
#is_jiamao                 #--  STRING COMMENT '存在售假嫌疑   #来源于wl_ind.whitebox_for_engine_full'   #
#is_zhijianxian            #--  STRING COMMENT '是否有商品质量鉴定险   #来源于dump的lasttable'   #
#-----------12-------------
pay_ord_cnt_3d_001_bin=[10**x for x in range(0,3)]         #--BIGINT COMMENT '最近3天成交笔数(支付子订单数)(剔除作弊订单)'   #
pay_ord_cnt_1w_001_bin=[10**x for x in range(0,5)]         #--BIGINT COMMENT '最近7天成交笔数(支付子订单数)(剔除作弊订单)'   #
pay_ord_cnt_2w_001_bin=[10**x for x in range(0,5)]         #--BIGINT COMMENT '最近15天成交笔数(支付子订单数)(剔除作弊订单)'   #
pay_ord_cnt_1m_001_bin=[10**x for x in range(0,6)]         #--BIGINT COMMENT '最近30天成交笔数(支付子订单数)(剔除作弊订单)'   #
pay_ord_cnt_2m_002_bin=[10**x for x in range(0,6)]         #--BIGINT COMMENT '最近60天成交笔数(支付子订单数)(剔除作弊订单)'   #
pay_ord_cnt_3m_001_bin=[10**x for x in range(0,7)]         #--BIGINT COMMENT '最近90天成交笔数(支付子订单数)(剔除作弊订单)'   #
dfns_cnt_6m_001_bin=[2**x-1 for x in range(0,10)]            #--BIGINT COMMENT '最近180天质量原因维权数   #来源于客满售后模型'   #
dfns_rate_6m_001_bin=vst_cart_byr_rate_1m_001_bin           #--DOUBLE COMMENT '最近180天质量原因维权占比   #来源于客满售后模型'   #
byr_rmk_cnt_6m_041_bin=[2**x for x in range(0,10)]         #--BIGINT COMMENT '最近180天4分以下描述DSR评价数   #来源于客满售后模型'   #
byr_rmk_rate_6m_003_bin=vst_cart_byr_rate_1m_001_bin         #--DOUBLE COMMENT '最近180天4分以下描述DSR评价占比   #来源于客满售后模型'   #
byr_rmk_cnt_6m_040_bin=[2**x-1 for x in range(0,10)]         #--BIGINT COMMENT '最近180天质量原因负面文本评价数   #来源于客满售后模型'   #
byr_rmk_rate_6m_002_bin=vst_cart_byr_rate_1m_001_bin        #--DOUBLE COMMENT '最近180天质量原因负面文本评价数占比   #来源于客满售后模型'   #
byr_rmk_byr_cnt_6m_003_bin=[2**x-1 for x in range(0,10)]     #--BIGINT COMMENT '最近180天质量原因负面文本评价买家数   #来源于客满售后模'   #
byr_rmk_byr_cnt_6m_002_bin=[2**x-1 for x in range(0,9)]     #--BIGINT COMMENT '最近180天质量原因极度负面文本评价买家数   #来源于客满售后模型'   #
inferior_index_bin=[x/10.0 for x in range(1,10)]             #--DOUBLE COMMENT '劣质指数   #来源于客满售后模型'   #
non_price_inferior_index_bin=inferior_index_bin   #--DOUBLE COMMENT '不考虑价格因素的劣质指数   #来源于客满售后模型'   #
itm_prc_6m_001_bin=vst_cart_byr_rate_1m_001_bin             # --DOUBLE COMMENT '最近180天低于同款中位价比例   #来源于客满售后模型'   #
#-- 2tiao_score                #--DOUBLE COMMENT '二跳分   #来源于search_offline.item_satisfaction_score的满意度分'   #
#--------13,在此处都是我们自己算出的图片的分，不是商品主图--------------------
#niupixian_bin=[0.3]                 #--  DOUBLE COMMENT '商品主图牛皮癣分   #判断商品主图上是否有牛皮癣   #包括水印文字等'   #
#facenum_bin=[0,1,2]                  #--  BIGINT COMMENT '商品主图人脸个数   #判断商品主图上人脸的个数'   #
#is_wbg_bin=[]                       #--  STRING COMMENT '商品主图是否白底   #判断商品主图商品主体之外的背景是否是白色'   #
#purebg_bin=[0.2]                     #--  DOUBLE COMMENT '商品主图纯色分   #判断商品主图商品主体之外的背景的复杂程度，背景越复杂纯度越低'   #
#mosaic_bin=[0.1]                     #--  DOUBLE COMMENT '商品主图是否包含多个主体   #判断商品主图是否包含多个商品或者多个模特，目前只支持男女装类目'   #
#objcolor_bin=[]                   #--  STRING COMMENT '商品主图主体位置及颜色   #判断商品主体位置以及主体的颜色'   #
#--------14--------------------
#pay_ord_amt_1w_001_bin=[10**x for x in range(0,8)]          #-- DOUBLE COMMENT '最近7天支付金额'   #
pay_ord_pbt_1w_001_bin=[10**x for x in range(0,8)]          #-- DOUBLE COMMENT '最近7天支付客单价'   #
pay_ord_itm_qty_1w_002_bin=[2**x-1 for x in range(0,11)]      #-- BIGINT COMMENT '最近7天支付商品件数'   #
#pay_ord_amt_1m_002_bin=[10**x for x in range(0,8)]          #-- DOUBLE COMMENT '最近30天支付金额'   #
pay_ord_pbt_1m_004_bin=[10**x for x in range(0,8)]          #-- DOUBLE COMMENT '最近30天支付客单价'   #
pay_ord_itm_qty_1m_001_bin=[2**x-1 for x in range(0,20)]      #-- BIGINT COMMENT '最近30天支付商品件数'   #
#pay_ord_amt_3m_001_bin=[10**x for x in range(0,8)]          #-- DOUBLE COMMENT '最近90天支付金额'   #
pay_ord_pbt_3m_001_bin=[10**x for x in range(0,8)]          #-- DOUBLE COMMENT '最近90天支付客单价'   #
pay_ord_byr_cnt_3m_001_bin=[2**x-1 for x in range(0,20)]      #-- BIGINT COMMENT '最近90天支付买家数'   #
#pay_ord_amt_1y_001_bin=[]          #-- DOUBLE COMMENT '最近365天支付金额'   #
#pay_ord_cnt_1y_001_bin=[2**x-1 for x in range(0,25)]          #-- BIGINT COMMENT '最近365天支付子订单数'   #
#pay_ord_itm_qty_1y_001_bin=[2**x-1 for x in range(0,25)]      #-- BIGINT COMMENT '最近365天支付商品件数'   #


@annotate('*->string,string,string,string,string,string,string,string')
class get_item_discrete_feature(BaseUDTF):

    def _init_(self):
        pass
    def process(self,
        item_id,title,cate_id,cate_level1_id,
        shop_id,org_brand_id,seller_id,
        #-------2----------
        lst_online_time,
        se_lpv_pc_1d_001,
        se_ipv_pc_1d_001,
        se_lpv_pc_1m_001,
        expo_uv_1m_001,
        clk_cnt_1m_015,
        clk_uv_1m_004,
        #-----3
        vst_cart_byr_rate_1m_001,
        vst_clt_byr_rate_1m_001,
        cart_cnt_3m_002,
        #-------4
        vst_pay_byr_rate_1d_001,
        pay_ord_byr_cnt_3d_001,
        vst_pay_byr_rate_3d_001,
        vst_pay_byr_rate_1w_017,
        pay_ord_byr_cnt_2w_001,
        vst_pay_byr_rate_2w_001,
        vst_pay_byr_rate_1m_005,
        #------5
        pay_ord_cnt_1d_015,
        pay_ord_cnt_1m_043,
        pay_ord_byr_cnt_1m_076,
        vst_crt_byr_rate_1m_002,
        vst_crt_byr_rate_1m_001,
        byr_rmk_rate_1m_003,
        byr_rmk_rate_1m_002,
        #-------6
        ipv_uv_3d_001,
        ipv_uv_2w_001,
        ipv_1d_001,
        ipv_uv_1d_001,
        ipv_3m_003,
        ipv_1w_001,
        ipv_1m_001,
        ipv_uv_1w_001,
        ipv_uv_1m_002,
        #-------7
        byr_rmk_cnt_6m_004,
        byr_rmk_cnt_1m_004,
        byr_rmk_cnt_1m_003,
        byr_rmk_cnt_1m_001,
        #----8
        clt_cnt_3d_001,
        clt_cnt_1w_002,
        clt_cnt_2w_001,
        clt_cnt_1m_002,
        clt_cnt_2m_003,
        clt_cnt_3m_003,
        clt_cnt_3m_004,
        clt_byr_cnt_1m_016,
        #------9
        cart_cnt_3d_001,
        cart_cnt_1w_001,
        cart_cnt_2w_001,
        cart_cnt_1m_001,
        cart_cnt_2m_001,
        cart_cnt_3m_001,
        cart_byr_cnt_1m_001,
        #-----10
        pay_ord_byr_cnt_1d_001,
        pay_ord_byr_cnt_1w_001,
        pay_ord_cnt_3m_004,
        pay_ord_byr_cnt_1m_002,
        pay_ord_itm_qty_3m_001,
        apl_rfd_cnt_1m_001,
        dspt_rfd_cnt_1m_001,
        dspt_rfd_rate_1m_001,
        #------11
        is_baojianpin,
        is_diaopai,
        is_fangxintao,
        is_foodqs,
        is_guanfangzhishou,
        is_haiwaizhiyou,
        is_huazhuangpin,
        is_huodaofukuan,
        is_jiajubaihuo,
        is_mianfeihuanxin,
        is_pinpaishouquan,
        is_pinzhichengnuo,
        is_poshunbuji,
        is_qiyemaijia,
        is_quanqiugou,
        is_rzershouche,
        is_tianmaoguoji,
        is_tuihuoyunfei,
        is_xinpin,
        is_yiyaozizhi,
        is_zhichianzhuang,
        is_zhongguozhizao,
        is_baoyou,
        is_jiamao,
        is_zhijianxian,
        #-------12
        pay_ord_cnt_3d_001,
        pay_ord_cnt_1w_001,
        pay_ord_cnt_2w_001,
        pay_ord_cnt_1m_001,
        pay_ord_cnt_2m_002,
        pay_ord_cnt_3m_001,
        dfns_cnt_6m_001,
        dfns_rate_6m_001,
        byr_rmk_cnt_6m_041,
        byr_rmk_rate_6m_003,
        byr_rmk_cnt_6m_040,
        byr_rmk_rate_6m_002,
        byr_rmk_byr_cnt_6m_003,
        byr_rmk_byr_cnt_6m_002,
        inferior_index,
        non_price_inferior_index,
        itm_prc_6m_001,
        #2tiao_score,
        #-----13
        #niupixian,
        #facenum,
        #is_wbg,
        #purebg,
        #mosaic,
        #objcolor,
        #---------14
        #pay_ord_amt_1w_001,
        pay_ord_pbt_1w_001,
        pay_ord_itm_qty_1w_002,
        #pay_ord_amt_1m_002,
        pay_ord_pbt_1m_004,
        pay_ord_itm_qty_1m_001,
        #pay_ord_amt_3m_001,
        pay_ord_pbt_3m_001,
        pay_ord_byr_cnt_3m_001
        #pay_ord_amt_1y_001,
        #pay_ord_cnt_1y_001,
        #pay_ord_itm_qty_1y_001
        ):
        v_str_list=[
        #-------2----------
        'lst_online_time',
        'se_lpv_pc_1d_001',
        'se_ipv_pc_1d_001',
        'se_lpv_pc_1m_001',
        'expo_uv_1m_001',
        'clk_cnt_1m_015',
        'clk_uv_1m_004',
        #-----3
        'vst_cart_byr_rate_1m_001',
        'vst_clt_byr_rate_1m_001',
        'cart_cnt_3m_002',
        #-------4
        'vst_pay_byr_rate_1d_001',
        'pay_ord_byr_cnt_3d_001',
        'vst_pay_byr_rate_3d_001',
        'vst_pay_byr_rate_1w_017',
        'pay_ord_byr_cnt_2w_001',
        'vst_pay_byr_rate_2w_001',
        'vst_pay_byr_rate_1m_005',
        #------5
        'pay_ord_cnt_1d_015',
        'pay_ord_cnt_1m_043',
        'pay_ord_byr_cnt_1m_076',
        'vst_crt_byr_rate_1m_002',
        'vst_crt_byr_rate_1m_001',
        'byr_rmk_rate_1m_003',
        'byr_rmk_rate_1m_002',
        #-------6
        'ipv_uv_3d_001',
        'ipv_uv_2w_001',
        'ipv_1d_001',
        'ipv_uv_1d_001',
        'ipv_3m_003',
        'ipv_1w_001',
        'ipv_1m_001',
        'ipv_uv_1w_001',
        'ipv_uv_1m_002',
        #-------7
        'byr_rmk_cnt_6m_004',
        'byr_rmk_cnt_1m_004',
        'byr_rmk_cnt_1m_003',
        'byr_rmk_cnt_1m_001',
        #----8
        'clt_cnt_3d_001',
        'clt_cnt_1w_002',
        'clt_cnt_2w_001',
        'clt_cnt_1m_002',
        'clt_cnt_2m_003',
        'clt_cnt_3m_003',
        'clt_cnt_3m_004',
        'clt_byr_cnt_1m_016',
        #------9
        'cart_cnt_3d_001',
        'cart_cnt_1w_001',
        'cart_cnt_2w_001',
        'cart_cnt_1m_001',
        'cart_cnt_2m_001',
        'cart_cnt_3m_001',
        'cart_byr_cnt_1m_001',
        #-----10
        'pay_ord_byr_cnt_1d_001',
        'pay_ord_byr_cnt_1w_001',
        'pay_ord_cnt_3m_004',
        'pay_ord_byr_cnt_1m_002',
        'pay_ord_itm_qty_3m_001',
        'apl_rfd_cnt_1m_001',
        'dspt_rfd_cnt_1m_001',
        'dspt_rfd_rate_1m_001',
        #------11
        'is_baojianpin',
        'is_diaopai',
        'is_fangxintao',
        'is_foodqs',
        'is_guanfangzhishou',
        'is_haiwaizhiyou',
        'is_huazhuangpin',
        'is_huodaofukuan',
        'is_jiajubaihuo',
        'is_mianfeihuanxin',
        'is_pinpaishouquan',
        'is_pinzhichengnuo',
        'is_poshunbuji',
        'is_qiyemaijia',
        'is_quanqiugou',
        'is_rzershouche',
        'is_tianmaoguoji',
        'is_tuihuoyunfei',
        'is_xinpin',
        'is_yiyaozizhi',
        'is_zhichianzhuang',
        'is_zhongguozhizao',
        'is_baoyou',
        'is_jiamao',
        'is_zhijianxian',
        #-------12
        'pay_ord_cnt_3d_001',
        'pay_ord_cnt_1w_001',
        'pay_ord_cnt_2w_001',
        'pay_ord_cnt_1m_001',
        'pay_ord_cnt_2m_002',
        'pay_ord_cnt_3m_001',
        'dfns_cnt_6m_001',
        'dfns_rate_6m_001',
        'byr_rmk_cnt_6m_041',
        'byr_rmk_rate_6m_003',
        'byr_rmk_cnt_6m_040',
        'byr_rmk_rate_6m_002',
        'byr_rmk_byr_cnt_6m_003',
        'byr_rmk_byr_cnt_6m_002',
        'inferior_index',
        'non_price_inferior_index',
        'itm_prc_6m_001',
        #2tiao_score',
        #-----13
        #niupixian',
        #facenum',
        #is_wbg',
        #purebg',
        #mosaic',
        #objcolor',
        #---------14
        #pay_ord_amt_1w_001',
        'pay_ord_pbt_1w_001',
        'pay_ord_itm_qty_1w_002',
        #pay_ord_amt_1m_002',
        'pay_ord_pbt_1m_004',
        'pay_ord_itm_qty_1m_001',
        #pay_ord_amt_3m_001',
        'pay_ord_pbt_3m_001',
        'pay_ord_byr_cnt_3m_001'
        #pay_ord_amt_1y_001',
        #pay_ord_cnt_1y_001',
        #pay_ord_itm_qty_1y_001
        ]
        
        feats=""
        bin="_bin"
        cln=":"
        sep="\001"
        #商品的叶子类目
        v_str="cate_id"
        v_str=v_str.strip()
        feats+=v_str+cln+str(eval(v_str))+sep
        #---------2---
        for v_str in v_str_list:
            v_str=v_str.strip()
            if v_str.startswith("is_"):#布尔值
                index=get_bool_index_by_value(eval(v_str),eval(v_str+bin))
            else:
                index=get_index_by_value(eval(v_str),eval(v_str+bin))
            feats+=v_str+cln+str(index)+sep
        feats=feats.rstrip(sep)
        if True:sys.stderr.write("output feats:%s\n"%(feats))
        self.forward(item_id,title,cate_id,cate_level1_id,
                     shop_id,org_brand_id,seller_id,feats)
        
        def get_index_by_value(self,value,list):
            if value is None or len(str(value))==0 or len(list)==0:
                return -1  #如果为null就是-1
            ind=0
            for x in list:
               if value<=x:return ind 
               else:ind+=1
            return ind #超出范围了
        
        def get_bool_index_by_value(self,value):
            if value is None or len(str(value))==0:
                return -1  #如果为null就是-1
            return value