package com.hankcs.algorithm;
// blog:http://www.cnblogs.com/skyme/p/4651331.html

/**
 * 维特比算法
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
        double[][] V = new double[obs.length][states.length]; // 5*2
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

            for (int cur_state : states)
            {
                double prob = -1;
                int max_prob_state;
                for (int pre_state : states)
                {
                    double nprob = V[t - 1][pre_state] * trans_p[pre_state][cur_state] * emit_p[cur_state][obs[t]];
                    if (nprob > prob)
                    {
                        prob = nprob;
                        max_prob_state = pre_state;
                        // 记录最大概率
                        V[t][cur_state] = prob;
                        // 记录路径
                        System.arraycopy(path[max_prob_state], 0, newpath[cur_state], 0, t);
                        newpath[cur_state][t] = cur_state;
                    }
                }
            }
            path = newpath;
        }

        double prob = -1;
        int state = 0;
        for (int y : states)
        {
            if (V[obs.length - 1][y] > prob)
            {
                prob = V[obs.length - 1][y];
                state = y;
            }
        }

        return path[state];
    }
}