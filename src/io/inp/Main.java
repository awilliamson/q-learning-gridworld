package io.inp;

public class Main {

    public static void main(String[] args) {
        // double init_epsilon, double final_epsilon, int episodes, double alpha, double gamma.
        QLearning qLearn = new QLearning( 0.1, 0.0001, 50000, 0.5, 0.9 );
        qLearn.run();

        System.out.print( "\n\n\n\n\n" );

        System.out.println( "Final" );
        qLearn.printMaxQValues();
        qLearn.printPolicy();

        System.out.println( "----------" );
        System.out.println( "X" + "\t" + "X" + "\t" + "G" );
        System.out.println( "S" + "\t" + "X" + "\t" + "X" );
        System.out.println( "X" + "\t" + "X" + "\t" + "T" );
        System.out.println( "----------" );
    }
}
