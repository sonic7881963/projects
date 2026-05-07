// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
import main from "./mian"
import enemy from './enemy';



const {ccclass, property} = cc._decorator;

@ccclass
export default class leftToRight extends cc.Component {

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
       
        if(main.attack == true){
            main.minion1_hp -= 30;
            let damage = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage.getComponent(cc.Label).string = "-30";
            let damage2 = cc.find("Canvas/bj/kuan/enemy_damage");
            damage2.getComponent(cc.Label).string = "";
        }
        
        main.attack = false;
        
    }


    onCollisionStay(other) {
        main.attack = false;
        main.minion_attack = false;
    }

    onCollisionExit(other) {
       cc.log("碰撞结束");
       
       
      
      
       if(main.minion1_hp <= 0) {   
        other.node.active = false;
        main.attack= false;
        main.minion_attack = false;
        main.main_hp= 163;
        main.minion1_hp= 163;
        main.minion2_hp= 163;
        main.minion3_hp = 163;
        cc.director.loadScene("fight2");
       } else {
        other.node.children[0].setContentSize(main.minion1_hp, 19);
       }
       
    }
    
    start () {
      
    this.main_x = this.node.position.x;
       
     
    }

    update (dt) {
        if (main.attack ==  true) {
         this.node.setPosition(this.node.position.x + 1000*dt, this.node.position.y);
            
        
        } else {
            if (!(this.node.position.x <= this.main_x + 50) && (this.node.position.x >= this.main_x - 50))   {
            this.node.setPosition(this.node.position.x - 1000*dt, this.node.position.y);
            }
            
        }

        leftToRight.current_x = this.node.position.x;
        leftToRight.current_y = this.node.position.y;
    }
    
}


