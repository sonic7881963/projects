// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
import global from "./global - 003"




const {ccclass, property} = cc._decorator;

@ccclass
export default class leftToRight_1 extends cc.Component {

    @property(cc.Label)
    label: cc.Label = null;

    @property
    text: string = 'hello';
    main_x: number;
    
    public static current_x: number;
    public static current_y: number;

    // LIFE-CYCLE CALLBACKS:

    onLoad () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    }
    

    //产生碰撞会调用
    onCollisionEnter(other,self){
      
       cc.log(other.node.name);
        if(global.attack == true){
            let damage = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage.getComponent(cc.Label).string = "-30";
            let damage2 = cc.find("Canvas/bj/kuan/enemy_damage");
            damage2.getComponent(cc.Label).string = "";
           if(other.node.name == "小鬼") {
            global.minion1_hp -= 30;
           } else  if(other.node.name == "小鬼2") {
            global.minion2_hp -= 30;
           } else  if(other.node.name == "小鬼3") {
            global.minion3_hp -= 30;
           }
        }
        
        global.attack = false;
    }


    onCollisionStay(other) {
        global.attack = false;
        global.minion_attack = false;
    }

    onCollisionExit(other) {

       
       
     
      
       if((global.minion1_hp <= 0)&&(other.node.name == "小鬼")) {   
        other.node.active = false;
       }  else if((global.minion2_hp <= 0)&&(other.node.name == "小鬼2")) {   
        other.node.active = false;
       } else if((global.minion3_hp <= 0)&&(other.node.name == "小鬼3")) {   
        other.node.active = false;
        cc.director.loadScene("fight8");
       } else {
           if(other.node.name == "小鬼") {
            other.node.children[0].setContentSize(global.minion1_hp, 19);
           } else  if(other.node.name == "小鬼2") {
            other.node.children[0].setContentSize(global.minion2_hp, 19);
           } else  if(other.node.name == "小鬼3") {
            other.node.children[0].setContentSize(global.minion3_hp, 19);
           }
       }
       
    }
    
    start () {
      
    this.main_x = this.node.position.x;
       
     
    }

    update (dt) {
        if (global.attack ==  true) {
         this.node.setPosition(this.node.position.x + 1000*dt, this.node.position.y);
            
        
        } else {
            if (!(this.node.position.x <= this.main_x + 50) && (this.node.position.x >= this.main_x - 50))   {
            this.node.setPosition(this.node.position.x - 1000*dt, this.node.position.y);
            }
            
        }

     
    }
    
}


