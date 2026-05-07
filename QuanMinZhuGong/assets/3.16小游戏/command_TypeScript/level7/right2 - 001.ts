
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html

const {ccclass, property} = cc._decorator;
import global from "./global - 003"



@ccclass
export default class enemy_2 extends cc.Component {
   
    // LIFE-CYCLE CALLBACKS:
    minion_x;
    
     count = 0;
    onLoad () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;

    }
    

    start () {
        this.schedule(() => {
            global.minion_attack = true;
        },1);

        this.minion_x = this.node.position.x;
    }
    
    onCollisionEnter(other,self){
       
        if(global.minion_attack == true) {
            global.main_hp -= 5;
            let damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-5";
            let damage2 = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage2.getComponent(cc.Label).string = "";
        }
        
       
    }

    onCollisionExit(other) {
    
        
      
        global.minion_attack = false;
       if( global.main_hp <= 0) {
        other.node.active = false;
        let lose = cc.find("Canvas/bj/fail");
        lose.active = true;
       } else {
        other.node.children[0].setContentSize( global.main_hp, 19);
       }
       
    }


    update (dt) {
      
        let node1 = cc.find("Canvas/bj/kuan/小鬼");
        if (this.count == 0) {
            if (node1.active == false) {
                let node2 = cc.find ("Canvas/bj/b/a")
                node2.active = true;
                node2.setContentSize(200,26);
                this.node.setPosition(  this.node.position.x , node1.position.y);   
                this.count++;        
            }
        }
        
        if (node1.active == false) {
            if (global.minion_attack ==  true) {
                this.node.setPosition(  this.node.position.x  - 1000*dt, this.node.position.y );
                   
               
               } else {
                   if (!(this.node.position.x >=  this.minion_x + 50) && (this.node.position.x <=  this.minion_x - 50))   {
                   this.node.setPosition(this.node.position.x + 1000*dt, this.node.position.y);
                   }
                   
               }
        }

        
    }
}
