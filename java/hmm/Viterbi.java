package com.hankcs.algorithm;
// blog:http://www.cnblogs.com/skyme/p/4651331.html

/**
 * 维特比算法
 *
 * 维特比算法就是求解HMM上的最短路径（-log(prob)，也即是最大概率）的算法, Viterbi被广泛应用到分词，词性标注等应用场景。
 *
 *
 * 求解最可能的隐状态序列是HMM的三个典型问题之一，通常用维特比算法解决。维特比算法就是求解HMM上的最短路径（-log(prob)，也即是最大概率）的算法。

 稍微用中文讲讲思路，

 1.很明显，第一天天晴还是下雨可以算出来：
 定义V[时间][今天天气] = 概率，注意今天天气指的是，前几天的天气都确定下来了（概率最大）今天天气是X的概率，这里的概率就是一个累乘的概率了。

 因为第一天我的朋友去散步了，所以第一天下雨的概率V[第一天][下雨] = 初始概率[下雨] * 发射概率[下雨][散步] = 0.6 * 0.1 = 0.06，同理可得V[第一天][天晴] = 0.24 。从直觉上来看，因为第一天朋友出门了，她一般喜欢在天晴的时候散步，所以第一天天晴的概率比较大，数字与直觉统一了。


 2. 从第二天开始，对于每种天气Y，都有前一天天气是X的概率 * X转移到Y的概率 * Y天气下朋友进行这天这种活动的概率。因为前一天天气X有两种可能，所以Y的概率有两个，选取其中较大一个作为V[第二天][天气Y]的概率，同时将今天的天气加入到结果序列中


 3. 比较V[最后一天][下雨]和[最后一天][天晴]的概率，找出较大的哪一个对应的序列，就是最终结果。

 算法的代码可以在github上看到，地址为：

 https://github.com/hankcs/Viterbi
 *
 * @author hankcs
 */
public class Viterbi
{
    /**
     * 求解HMM模型
     * @param obs 观测序列
     * @param states 隐状态
     * @param start_p 初始概率（隐状态）
     * @param trans_p 转移概率（隐状态）
     * @param emit_p 发射概率 （隐状态表现为显状态的概率）
     * @return 最可能的序列
     */
    public static int[] compute(int[] obs, int[] states, double[] start_p, double[][] trans_p, double[][] emit_p)
    {
        // 观测到此序列的概率矩阵
        double[][] V = new double[obs.length][states.length]; // 5*2, 观测序列矩阵的概率
        int[][] path = new int[states.length][obs.length]; // 2*5

        for (int state : states)
        {
            // 初始状态概率*在此状态下的发射概率
            V[0][state] = start_p[state] * emit_p[state][obs[0]];
            path[state][0] = state;
        }

        // 从下一个状态（index=1）开始
        for (int t = 1; t < obs.length; ++t)
        {
            int[][] newpath = new int[states.length][obs.length]; // 2*5
            // 对于当前的每个状态，计算一个最大的概率
            for (int cur_state : states)
            {
                double max_trainsfer_prob = -1.0;
                int max_prob_pre_state;
                // 当前每个状态的概率，依赖于前一个状态的概率
                for (int pre_state : states)
                {
                    // 计算最大的转移概率
                    double trainsfer_prob = V[t - 1][pre_state] * trans_p[pre_state][cur_state];
                    if (trainsfer_prob > max_trainsfer_prob)
                    {
                        max_trainsfer_prob = trainsfer_prob;
                        max_prob_pre_state = pre_state;
                        // 记录最大概率 = 转移*发射
                        V[t][cur_state] = max_trainsfer_prob* emit_p[cur_state][obs[t]];
                        // 记录路径
                        System.arraycopy(path[max_prob_pre_state], 0, newpath[cur_state], 0, t); // 将path[a]这一行的值拷到path[b]中
                        newpath[cur_state][t] = cur_state;
                    }
                }
            }
            path = newpath;
        }

        // 找出最后一天里，概率最大的状态即为最终结果
        double max_prob = -1;
        int state = 0;
        for (int y : states)
        {
            if (V[obs.length - 1][y] > max_prob)
            {
                max_prob = V[obs.length - 1][y];
                state = y;
            }
        }
        System.out.println(String.format("prob:%.8f",max_prob));
        return path[state];
    }
}