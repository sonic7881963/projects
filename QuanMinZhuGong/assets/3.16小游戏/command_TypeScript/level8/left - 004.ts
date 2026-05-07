// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
import gloabl from "./global - 004"




const {ccclass, property} = cc._decorator;

@ccclass
export default class mainCharacter extends cc.Component {

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
        cc.log("开始碰撞"+other.tag);
       
        if(gloabl.attack == true){
            gloabl.minion1_hp -= 999;
            let damage = cc.find("Canvas/bj/kuan/main_damage");
            damage.getComponent(cc.Label).string = "-999";
            let damage2 = cc.find("Canvas/bj/kuan/enemy_damage");
            damage2.getComponent(cc.Label).string = "";
        }
        
        gloabl.attack = false;
    }


    onCollisionStay(other) {
        gloabl.attack = false;
        gloabl.minion_attack = false;
    }

    onCollisionExit(other,self) {
       cc.log("碰撞结束");
       if(gloabl.minion1_hp <= 0) {   
        other.node.active = false;
        gloabl.attack= false;
        gloabl.minion_attack = false;
        let damage = cc.find("Canvas/bj/kuan/main_damage");
        damage.getComponent(cc.Label).string = "";
        let damage2 = cc.find("Canvas/bj/kuan/enemy_damage");
        damage2.getComponent(cc.Label).string = "";
        let node1 = cc.find("Canvas/bj/d-8/c-8");
        node1.active = true;
        let node2 = cc.find("Canvas/bj/b/a");
        node2.active = true;
        self.node.active = false;
        let node3 = cc.find("Canvas/bj/kuan/tgcg");
        node3.active = true;
       } else {
        other.node.children[0].setContentSize(gloabl.minion1_hp, 19);
       }
       
    }
    
    start () {
      
    this.main_x = this.node.position.x;
       
     
    }

    update (dt) {
        if (gloabl.attack ==  true) {
         this.node.setPosition(this.node.position.x + 1000*dt, this.node.position.y);
            
        
        } else {
            if (!(this.node.position.x <= this.main_x + 50) && (this.node.position.x >= this.main_x - 50))   {
            this.node.setPosition(this.node.position.x - 1000*dt, this.node.position.y);
            }
            
        }

    }
    
}


