package io.inp;

public class Main {

    public static void main(String[] args) {
        // double init_epsilon, double final_epsilon, int explore, int episodes, double gamma
        QLearning qLearn = new QLearning( 0.001, 0.00001, 50000, 1000000, 0.99 );
        qLearn.run();

        System.out.println( "Final" );
        qLearn.printMaxQValues();
        qLearn.printPolicy();
        System.out.println( "-----" );
    }
}
