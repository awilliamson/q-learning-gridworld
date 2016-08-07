package io.inp;

public class Main {

    public static void main(String[] args) {
        // double init_epsilon, double final_epsilon, int episodes, double alpha, double gamma.
        QLearning qLearn = new QLearning( 0.1, 0.0001, 10000000, 0.05, 0.9 );
        qLearn.run();

        System.out.println( "Final" );
        qLearn.printMaxQValues();
        qLearn.printPolicy();
        System.out.println( "-----" );
    }
}
