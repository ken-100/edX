import java.util.Arrays;

class RedAstronaut extends Player implements Impostor {
    private String skill;
    public RedAstronaut(String name) {
        this(name, 15, "experienced");
    }
    public RedAstronaut(String name, int susLevel, String skill) {
        super(name, susLevel);
        this.skill = skill;
    }
    public void emergencyMeeting() {
        if (!this.isFrozen()) {
            Player[] players = Player.getPlayers();
            Arrays.sort(players);

            int mostSusIndex = players.length - 1;
            Player mostSusPlayer = players[mostSusIndex]; 

            while (mostSusIndex >= 0) {
                mostSusPlayer = players[mostSusIndex];
                if (!mostSusPlayer.isFrozen() && mostSusPlayer != this) {
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
    public void freeze(Player p) {
        if (p instanceof Impostor) {
            return;
        }
        if (this.isFrozen()) {
            return;
        }
        if (p.isFrozen()){
            return;
        }
        if (this.getSusLevel() < p.getSusLevel()) {
            // successful freeze
            p.setFrozen(true);
        } else {
            // unsuccessful freeze
            this.setSusLevel(this.getSusLevel() * 2);
        }
        this.gameOver();
    }
    public void sabotage(Player p) {
        if (p instanceof Impostor) {
            return;
        }
        if (this.isFrozen()) {
            return;
        }
        if (p.isFrozen()) {
            return;
        }
        if (this.getSusLevel() < 20) {
            p.setSusLevel((int) (p.getSusLevel() * 1.5));
        } else {
            p.setSusLevel((int) (p.getSusLevel() * 1.25));
        }
    }
    public boolean equals(Object o) {
        if (!(o instanceof RedAstronaut)) {
            return false;
        }
        RedAstronaut otherPlayer = (RedAstronaut) o;
        return otherPlayer.getName() == this.getName() && 
               otherPlayer.isFrozen() == this.isFrozen() && 
               otherPlayer.getSusLevel() == this.getSusLevel() &&
               otherPlayer.skill == this.skill; 
    }
    public String toString() {
        String playerString = super.toString();
        String redAstronautString =  " I am an " + this.skill + " player!";
        String finalString = playerString + redAstronautString;

        if (this.getSusLevel() > 15) {
            finalString = finalString.toUpperCase();
        }
        return finalString;
    }
    public String getSkill() {
        return this.skill;
    }
    public void setSkill(String skill) {
        this.skill = skill;
    }
}
