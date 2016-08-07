package io.inp;

public class Main {

    public static void main(String[] args) {
        // double init_epsilon, double final_epsilon, int explore, int episodes, double gamma
        QLearning qLearn = new QLearning( 0.1, 0.0001, 30000, 0.9 );
        qLearn.run();

        System.out.println( "Final" );
        qLearn.printMaxQValues();
        qLearn.printPolicy();
        System.out.println( "-----" );
    }
}
