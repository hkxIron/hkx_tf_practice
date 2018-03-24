package com.hankcs.algorithm;
import com.hankcs.algorithm.Viterbi;

public class DoctorExample
{
    enum Status
    {
        Healthy,
        Fever,
    }
    enum Feel
    {
        normal,
        cold,
        dizzy,
    }

    static int[] states = new int[]{Status.Healthy.ordinal(), Status.Fever.ordinal()};
    static int[] observations = new int[]{Feel.normal.ordinal(), Feel.cold.ordinal(), Feel.dizzy.ordinal()};
    static double[] start_probability = new double[]{0.6, 0.4};
    static double[][] transititon_probability = new double[][]{
            {0.7, 0.3},
            {0.4, 0.6},
    };
    static double[][] emission_probability = new double[][]{
            {0.5, 0.4, 0.1},
            {0.1, 0.3, 0.6},
    };

    public static void main(String[] args)
    {
        // 用维特比算法，解码隐状态序列
        int[] result = Viterbi.compute(observations, states, start_probability, transititon_probability, emission_probability);
        for (int r : result)
        {
            System.out.print(Status.values()[r] + " ");
        }
        System.out.println();
    }
}