import java.util.Arrays;

public class BlueAstronaut extends Player implements Crewmate {
    private int numTasks;
    private int taskSpeed;
    public BlueAstronaut(String name) {
        this(name, 15, 6, 10);
    }
    public BlueAstronaut(String name, int susLevel, int numTasks, int taskSpeed) {
        super(name, susLevel);
        this.numTasks = numTasks;
        this.taskSpeed = taskSpeed;
    }
    public void emergencyMeeting() {
        if (!this.isFrozen()) {
            Player[] players = Player.getPlayers();
            Arrays.sort(players);

            int mostSusIndex = players.length - 1;
            Player mostSusPlayer = players[mostSusIndex]; 

            while (mostSusIndex >= 0) {
                mostSusPlayer = players[mostSusIndex];
                if (!mostSusPlayer.isFrozen()) {
                    break;
                }
                mostSusIndex--;
            }

            Player secondMostSusPlayer = players[mostSusIndex - 1];
            if (secondMostSusPlayer.getSusLevel() == mostSusPlayer.getSusLevel()) {
                return;
            } else {
                mostSusPlayer.setFrozen(true);
                mostSusPlayer.gameOver();
            }
        }
    }
    public void completeTask() {
        if (!this.isFrozen() && this.numTasks != 0) {
            if (this.taskSpeed > 20) {
                if (this.numTasks == 2 || this.numTasks == 1) {
                    this.setSusLevel((int) (this.getSusLevel() * 0.5));
                    System.out.println("I have completed all my tasks");
                }
                setNumTasks(this.numTasks - 2);
            } else {
                if (this.numTasks == 1) {
                this.setSusLevel((int) (this.getSusLevel() * 0.5));
                    System.out.println("I have completed all my tasks");
                }
                setNumTasks(this.numTasks - 1);
            }
        }
    }
    public void setNumTasks(int numTasks) {
        if (numTasks < 0) {
            this.numTasks = 0;
        } else {
            this.numTasks = numTasks;
        }
    }
    public boolean equals(Object o) {
        if (!(o instanceof BlueAstronaut)) {
            return false;
        }
        BlueAstronaut otherPlayer = (BlueAstronaut) o;

        return this.getName() == otherPlayer.getName() &&
               this.isFrozen() == otherPlayer.isFrozen() &&
               this.getSusLevel() == otherPlayer.getSusLevel() &&
               this.numTasks == otherPlayer.numTasks &&
               this.taskSpeed == otherPlayer.taskSpeed;
    }
    public String toString() {
        String playerString = super.toString();
        String blueAstronautString =  " I have " + this.numTasks + " left over.";
        String finalString = playerString + blueAstronautString;

        if (this.getSusLevel() > 15) {
            finalString = finalString.toUpperCase();
        }
        return finalString;
    }
    public int getNumTasks() {
        return this.numTasks;
    }
    public int getTaskSpeed() {
        return this.taskSpeed;
    }
    public void setTaskSpeed(int speed) {
        this.taskSpeed = speed;
    }
}
