package com.hankcs.algorithm;
import com.hankcs.algorithm.Viterbi;

public class WeatherExample
{
    enum Weather
    {
        Rainy,
        Sunny,
    }
    enum Activity
    {
        walk,
        shop,
        clean,
    }
    // 观测到的序列
    static int[] observations = new int[]{Activity.walk.ordinal(), Activity.shop.ordinal(), Activity.clean.ordinal(), Activity.shop.ordinal(),Activity.walk.ordinal()};

    static int[] states = new int[]{Weather.Rainy.ordinal(), Weather.Sunny.ordinal()}; // ordinal:为enum的下标

    // 初始状态概率
    static double[] start_probability = new double[]{0.6, 0.4};
    // 隐状态转移概率
    static double[][] transititon_probability = new double[][]{
            {0.7, 0.3},
            {0.4, 0.6},
    };
    // 发射概率，即每种状态下，采取不同活动的概率
    static double[][] emission_probability = new double[][]{
            {0.1, 0.4, 0.5},
            {0.6, 0.3, 0.1},
    };

    public static void main(String[] args)
    {
        // 用维特比算法，解码隐状态序列，根据观测到的序列，来推断其最可能的隐状态序列
        // 在已经知晓其活动的序列后，我们去推断当地的天气
        int[] result = Viterbi.compute(observations, states, start_probability, transititon_probability, emission_probability);
        for (int r : result)
        {
            System.out.print(Weather.values()[r] + " ");
        }
        System.out.println();
    }
}