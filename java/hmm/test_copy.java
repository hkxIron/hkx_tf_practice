/**
 *
 *
 *
 */

public class test_copy
{
    public static void print_arr(int[][]matrix){
        for(int[] vec:matrix){
            for(int e : vec) {
                System.out.print(e+" ");
            }
            System.out.println();
        }
    }

    public static void main(String[] args)
    {
        int[][] a = new int[2][3];
        a[0][0] = -5;
        a[0][1] = -1;
        a[0][2] = -2;
        a[1][1] = -4;
        a[1][2] = -3;
        print_arr(a);

        int[][] b = new int[2][3];
        System.arraycopy(a[0], 0, b[0], 0, 3); // 将a[0]这一行拷到b[0]那一行去

        print_arr(b);
    }
}