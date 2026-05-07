
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html

const {ccclass, property} = cc._decorator;
import gloabl from "./global - 004"



@ccclass
export default class right extends cc.Component {
   
    // LIFE-CYCLE CALLBACKS:
    minion_x;
    
    onLoad () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    }
    

    start () {
        this.schedule(() => {
            gloabl.minion_attack = true;
        },1);

        this.minion_x = this.node.position.x;
    }
    
    onCollisionEnter(other,self){
        cc.log("开始碰撞"+other.tag);
        if(gloabl.minion_attack == true) {
            let damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-100";
            let damage2 = cc.find("Canvas/bj/kuan/main_damage");
            damage2.getComponent(cc.Label).string = "";
            gloabl.main_hp -= 40;
        }
        
       
    }

    onCollisionExit(other) {
       cc.log("碰撞结束");
        
      
       gloabl.minion_attack = false;
       if( gloabl.main_hp <= 0) {
        other.node.active = false;
        let lose = cc.find("Canvas/bj/fail");
        lose.active = true;
       } else {
        other.node.children[0].setContentSize( gloabl.main_hp, 19);
       }
       
    }


    update (dt) {

        if (gloabl.minion_attack ==  true) {
            this.node.setPosition(  this.node.position.x  - 1000*dt, this.node.position.y );
               
           
           } else {
               if (!(this.node.position.x >=  this.minion_x + 50) && (this.node.position.x <=  this.minion_x - 50))   {
               this.node.setPosition(this.node.position.x + 1000*dt, this.node.position.y);
               }
               
           }
    }
}
