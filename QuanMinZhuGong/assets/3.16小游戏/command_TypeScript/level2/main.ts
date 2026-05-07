// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html

const {ccclass, property} = cc._decorator;

@ccclass
export default class main_1 extends cc.Component {

    @property(cc.Label)
    label: cc.Label = null;

    @property
    text: string = 'hello';

    // LIFE-CYCLE CALLBACKS:

    // onLoad () {}
   
   public static attack: boolean = false;
   public static minion_attack: boolean = false;
   public static main_hp: number = 163;
   public static minion1_hp: number = 163;
   public static minion2_hp: number = 163;
   public static minion3_hp: number = 163;
   

    start () {
        this.node.on('touchstart', this.onEventStart, this);
    }
    onEventStart() {
        main_1.attack = true;
       
    }
    update (dt) {
        cc.log("min1, min2, min3 "+main_1.minion1_hp+" "+main_1.minion3_hp+" "+main_1.minion3_hp)
    }
}
